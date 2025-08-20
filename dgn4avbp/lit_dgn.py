# Lightning wrapper for DiffusionGraphNet with CFDDataset sequences
# Drop this into your project and import `LitDiffusionCFD`.

from __future__ import annotations
import torch
import lightning as L
from torch import optim
from typing import List, Dict, Any
from torch_geometric.data import Data, Batch

# Assumptions:
# - Your diffusion model is `dgn.dgn_model.DiffusionGraphNet` (subclass of DiffusionModel)
# - Your loss callable matches: loss(model, graph) -> per-sample tensor [B]
# - CFDDataset returns a CombinedLoader that yields either:
#     (a) a List[Batch]  (sequence for one stream), or
#     (b) a Dict[str, List[Batch]] when combining multiple streams.
# - Each List[Batch] has length seq_len and each Batch is a PyG Batch with .target, .pos, .edge_index, etc.

# -------------------------
# 1) Sequence -> single graph adapter
# -------------------------

def _normalize_idx(idx, C, *, allow_empty=False, name="idx"):
    if idx is None:
        return [] if allow_empty else list(range(C))
    if isinstance(idx, range):
        idx = list(idx)
    elif isinstance(idx, slice):
        idx = list(range(C))[idx]
    elif isinstance(idx, torch.Tensor):
        if idx.numel() == 0:
            return []
        if idx.dtype not in (torch.long, torch.int64):
            raise TypeError(f"{name} must be integer indices, got dtype={idx.dtype}")
        idx = idx.detach().cpu().view(-1).tolist()
    elif isinstance(idx, (list, tuple)):
        idx = list(idx)
    else:
        raise TypeError(f"Unsupported {name} type: {type(idx)}")

    idx = sorted(set(int(i) for i in idx))
    if len(idx) == 0:
        if allow_empty:
            return []
        raise ValueError(f"{name} resolved to empty, but allow_empty=False.")
    lo, hi = min(idx), max(idx)
    if lo < 0 or hi >= C:
        raise IndexError(f"{name} out of bounds for target with C={C}. got min={lo}, max={hi}.")
    return idx

def split_target_into_y_and_cond(graph, y_idx, cond_idx, add_time=False):
    """
    Move selected columns from `graph.target` to:
      - `graph.target` := y (the supervised channels)
      - `graph.cond`   := conditions (from target and optionally time)
    """
    tgt = graph.target
    assert tgt.dim() == 2, f"target must be [N,C], got {tuple(tgt.shape)}"
    N, C = tgt.shape

    y_idx    = _normalize_idx(y_idx,    C, allow_empty=False, name="y_idx")
    cond_idx = _normalize_idx(cond_idx, C, allow_empty=True,  name="cond_idx")

    overlap = set(y_idx).intersection(cond_idx)
    if overlap:
        raise ValueError(f"y_idx and cond_idx overlap on columns {sorted(overlap)}.")

    dev = tgt.device
    tgt_cpu = tgt.detach().cpu()

    y    = tgt_cpu[:, y_idx].contiguous()
    cond = tgt_cpu[:, cond_idx].contiguous() if len(cond_idx) else None

    if add_time and hasattr(graph, "time"):
        tcol = graph.time.detach().cpu().view(-1, 1).float()
        cond = tcol if cond is None else torch.cat([cond, tcol], dim=1)

    graph.target = y.to(dev)
    if cond is not None:
        graph.cond = cond.to(dev)

    return graph

def inspect(obj, name="obj"):
    print(f"\n=== inspect: {name} (type={type(obj).__name__}) ===")
    if isinstance(obj, dict):
        print(f"Dict with keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"- {k}: type={type(v).__name__}")
    elif isinstance(obj, list):
        print(f"List length: {len(obj)}")
        if len(obj) > 0:
            print(f"First elem type: {type(obj[0]).__name__}")
    elif isinstance(obj, (Data, Batch)):
        for k, v in obj.items():
            shape  = getattr(v, "shape", None)
            device = getattr(v, "device", None)
            print(f"{k:12} -> {type(v).__name__} {shape} {device}")
    else:
        print(obj)

def sequence_to_graph_single_target(sequence, use_past_as_cond=False):
    graph = sequence[-1]                    # pick the last frame (a Batch)
    assert hasattr(graph, 'target'), "Each frame must carry node-wise 'target'"

    if use_past_as_cond and len(sequence) > 1:
        past = torch.cat([seq_t.target for seq_t in sequence[:-1]], dim=1)
        graph.cond = past
    else:
        if hasattr(graph, "cond"):
            graph.cond = None
    return graph

# --- NEW: choose a sub-window before packing -----------------------------------
def select_subsequence(sequence: List[Batch], win_len: int, stride: int = 1, policy: str = "random") -> List[Batch]:
    """
    Pick a temporal sub-window from `sequence`.
    Need = (win_len - 1) * stride + 1 frames.
    """
    L = len(sequence)
    need = (win_len - 1) * stride + 1
    if need > L:
        raise ValueError(f"Need {need} frames (win_len={win_len}, stride={stride}) but got {L}.")
    if policy == "random":
        start = torch.randint(0, L - need + 1, ()).item()
    elif policy == "last":
        start = L - need
    elif policy == "first":
        start = 0
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return sequence[start : start + need : stride]

# --- NEW: a general packer for temporal windows --------------------------------
def pack_time_window_graph(
    sequence: list[Batch],
    *,
    y_idx=None, cond_idx=None,        # <---
    y_cols: int | None = None,        # used only if y_idx is None
    cond_cols: int = 0,               # used only if cond_idx is None
    mode: str = "y_window_cond_static",
    include_time: bool = False,
) -> Batch:
    assert len(sequence) >= 1
    base = sequence[-1]
    dev  = base.target.device

    #y_steps, c_steps = []
    y_steps, c_steps = [], []
    for g in sequence:
        N, C = g.target.shape
        # y slice
        if y_idx is not None:
            yi = _normalize_idx(y_idx, C, allow_empty=False, name="y_idx")
            y_t = g.target[:, yi]
        else:
            assert y_cols is not None, "Provide y_idx or y_cols"
            assert C >= y_cols + cond_cols, "Per-step C smaller than y_cols+cond_cols"
            y_t = g.target[:, :y_cols]
        y_steps.append(y_t)
        # cond slice
        if cond_idx is not None:
            ci = _normalize_idx(cond_idx, C, allow_empty=True, name="cond_idx")
            c_t = g.target[:, ci] if len(ci) else None
        else:
            c_t = g.target[:, y_cols:y_cols+cond_cols] if (cond_cols > 0 and y_idx is None) else None
        if c_t is not None:
            c_steps.append(c_t)

    Y = torch.stack(y_steps, dim=1)  # [N, L, y_dim]
    N, L, y_dim = Y.shape

    if mode == "y_window_cond_static":
        base.target = Y.reshape(N, L * y_dim).contiguous().to(dev)
        if len(c_steps):
            base.cond = c_steps[-1].contiguous().to(dev)
        elif hasattr(base, "cond"):
            delattr(base, "cond")
    elif mode == "y_window_cond_window":
        base.target = Y.reshape(N, L * y_dim).contiguous().to(dev)
        if len(c_steps):
            Cseq = torch.stack(c_steps, dim=1)  # [N, L, cond_dim]
            base.cond = Cseq.reshape(N, -1).contiguous().to(dev)
        elif hasattr(base, "cond"):
            delattr(base, "cond")
    elif mode == "y_last_cond_past_y":
        assert L >= 2
        base.target = Y[:, -1, :].contiguous().to(dev)
        base.cond   = Y[:, :-1, :].reshape(N, (L - 1) * y_dim).contiguous().to(dev)
    else:
        raise ValueError(f"Unknown pack mode: {mode}")

    if include_time and hasattr(base, "time"):
        t = base.time.view(-1, 1).float().to(dev)
        base.cond = torch.cat([base.cond, t], dim=1) if hasattr(base, "cond") else t
    if hasattr(base, "time"):
        delattr(base, "time")
    return base


# -------------------------
# 2) LightningModule that wraps your DiffusionGraphNet and the DiffusionProcess
# -------------------------
class LitDiffusionCFD(L.LightningModule):
    def __init__(
        self,
        net,
        diffusion_process,
        criterion,
        step_sampler_factory,
        lr: float = 2e-4,
        scheduler_cfg: dict | None = None,
        use_past_as_cond: bool = False,    # still supported for the non-packed path
        # --- NEW: packing knobs ---
        pack_mode: str | None = None,      # None = don’t pack; otherwise one of the modes above
        y_cols: int = 6,
        cond_cols: int = 3,
        include_time_in_cond: bool = False,
        pack_win_len: int | None = None,   # e.g., 4
        pack_stride: int = 1,              # e.g., 2
        pack_select: str = "random",       # "random" | "last" | "first"
        y_idx=None,          # e.g. [2,0,5] or slice(0,6) or torch.LongTensor(...)
        cond_idx=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["net", "diffusion_process", "criterion", "step_sampler_factory"]
        )

        self.net = net
        self.net.diffusion_process = diffusion_process
        self.dp = diffusion_process
        self.criterion = criterion
        self.step_sampler = step_sampler_factory(num_diffusion_steps=self.dp.num_steps)
        self.lr = lr
        self.scheduler_cfg = scheduler_cfg

        # Old path toggle (kept for compatibility)
        self.use_past_as_cond = use_past_as_cond

        # NEW packing config
        self.pack_mode = pack_mode
        self.y_cols = y_cols
        self.cond_cols = cond_cols
        self.include_time_in_cond = include_time_in_cond
        self.pack_win_len = pack_win_len
        self.pack_stride = pack_stride
        self.pack_select = pack_select
        # NEW
        self.y_idx = y_idx
        self.cond_idx = cond_idx

    @staticmethod
    def _as_streams(batch: Any) -> List[List[Batch]]:
        if isinstance(batch, list):
            return [batch]
        if isinstance(batch, dict):
            return list(batch.values())
        raise TypeError(f"Unexpected batch type: {type(batch)}. Expected List[Batch] or Dict[str, List[Batch]].")

    def _prepare_graph(self, sequence: list[Batch]) -> Batch:
        if self.pack_mode:
            # pick a sub-window BEFORE packing
            seq = sequence
             
            if self.pack_win_len is not None:
                seq = select_subsequence(
                    sequence, win_len=self.pack_win_len,
                    stride=self.pack_stride, policy=self.pack_select  # "random"|"last"|"first"
                )

            graph = pack_time_window_graph(
                seq,
                # support explicit channel slices
                y_idx=self.y_idx, cond_idx=self.cond_idx,
                # or fall back to contiguous layout [y_cols | cond_cols]
                y_cols=self.y_cols, cond_cols=self.cond_cols,
                mode=self.pack_mode,
                include_time=self.include_time_in_cond,
            )
        else:
            graph = sequence_to_graph_single_target(sequence, use_past_as_cond=self.use_past_as_cond)

            # If you want past-as-cond, either:
            #  - skip passing cond_idx (leave cond from past), or
            #  - use pack_mode="y_last_cond_past_y"
            # Here we only split cond if cond_idx is provided.
            if self.y_idx is not None or self.cond_idx is not None:
                graph = split_target_into_y_and_cond(
                    graph,
                    y_idx = self.y_idx if self.y_idx is not None else range(self.y_cols),
                    cond_idx = [] if self.cond_idx is None else self.cond_idx,
                    add_time=self.include_time_in_cond,
                )
            else:
                # fallback to contiguous ranges
                graph = split_target_into_y_and_cond(
                    graph,
                    y_idx=range(self.y_cols),
                    cond_idx=range(self.y_cols, self.y_cols + self.cond_cols),
                    add_time=self.include_time_in_cond,
                )

        inspect(graph, "graph (prepared)")
        return graph

    # ---------------------
    # Standard Lightning hooks
    # ---------------------
    def forward(self, graph: Batch):  # inference-time forward (delegates to your model)
        return self.net(graph)

    def _one_graph_loss(self, graph: Batch) -> torch.Tensor:
        """Run one diffusion training pass on a prepared graph and return scalar loss."""
        device = self.device
        graph = graph.to(device)

        # Latent diffusion path if present
        if getattr(self.net, "is_latent_diffusion", False):
            graph = self.net.autoencoder.transform(graph)

        # Sample diffusion steps and weights per-graph in batch
        batch_size = graph.batch.max().item() + 1
        r, sample_weight = self.step_sampler(batch_size, device)
        graph.r = r

        # Diffuse x0 -> xt
        graph.field_start = graph.x_latent_target if getattr(self.net, "is_latent_diffusion", False) else graph.target
        graph.field_r, graph.noise = self.dp(
            field_start    = graph.field_start,
            r              = graph.r,
            batch          = graph.batch,
            dirichlet_mask = None if getattr(self.net, "is_latent_diffusion", False) else getattr(graph, 'dirichlet_mask', None),
        )

        # Per-sample loss, then weight and reduce
        per_sample = self.criterion(self.net, graph)  # [B]
        if hasattr(self.step_sampler, "update"):
            # Importance sampler update (no grad)
            self.step_sampler.update(graph.r, per_sample.detach())
        loss = (per_sample * sample_weight).mean()
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        streams = self._as_streams(batch)
        losses = []
        for seq in streams:
            graph = self._prepare_graph(seq)
            losses.append(self._one_graph_loss(graph))
        loss = torch.stack(losses).mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        streams = self._as_streams(batch)
        losses = []
        for seq in streams:
            graph = self._prepare_graph(seq)
            losses.append(self._one_graph_loss(graph))
        val_loss = torch.stack(losses).mean()
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return val_loss

    def configure_optimizers(self):
        opt = optim.Adam(self.net.parameters(), lr=self.lr)
        if self.scheduler_cfg:
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=self.scheduler_cfg.get("factor", 0.5),
                patience=self.scheduler_cfg.get("patience", 10),
                eps=0.0,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "train/loss_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return opt

    def on_after_backward(self):
        if not self.training or (hasattr(self, 'trainer') and self.trainer.sanity_checking):
            return
        total = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total += float(g.norm(2).item() ** 2)
        grad_norm = total ** 0.5
        self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=False)
