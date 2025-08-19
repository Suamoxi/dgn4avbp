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
from torch_geometric.data import Batch


from torch_geometric.data import Data, Batch

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
    inspect(sequence, "sequence")           # <- this is a list
    graph = sequence[-1]                    # pick the last frame (a Batch)
    inspect(graph, "graph (last frame)")    # <- now .items() works

    # Ensure mandatory fields
    assert hasattr(graph, 'target'), "Each frame must carry node-wise 'target'"

    # Optional: pack past frames into `cond` (disabled by default to keep feature sizes stable)
    if use_past_as_cond and len(sequence) > 1:
        past = torch.cat([seq_t.target for seq_t in sequence[:-1]], dim=1)
        graph.cond = past
    else:
        if hasattr(graph, "cond"):
            graph.cond = None

    # Do NOT touch graph.glob / graph.loc / graph.omega / edge_cond ... they are already read by the model
    return graph


# -------------------------
# 2) LightningModule that wraps your DiffusionGraphNet and the DiffusionProcess
# -------------------------
class LitDiffusionCFD(L.LightningModule):
    def __init__(
        self,
        net,                      # DiffusionGraphNet (DiffusionModel)
        diffusion_process,        # DiffusionProcess
        criterion,                # callable: (model, graph) -> per-sample loss [B]
        step_sampler_factory,     # callable factory: step_sampler(num_diffusion_steps=...)
        lr: float = 2e-4,
        scheduler_cfg: dict | None = None,  # e.g., {"factor":0.5, "patience":10}
        use_past_as_cond: bool = False,     # set True to concat past frames into .cond
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "diffusion_process", "criterion", "step_sampler_factory"])  

        # Core pieces
        self.net = net
        self.net.diffusion_process = diffusion_process
        self.dp = diffusion_process
        self.criterion = criterion
        self.step_sampler = step_sampler_factory(num_diffusion_steps=self.dp.num_steps)
        self.lr = lr
        self.scheduler_cfg = scheduler_cfg
        self.use_past_as_cond = use_past_as_cond

    # ---------------------
    # Helpers for CFDDataset batches
    # ---------------------
    @staticmethod
    def _as_streams(batch: Any) -> List[List[Batch]]:
        """Normalize incoming batch into a list of sequence streams.
        - If `batch` is List[Batch], return [batch].
        - If `batch` is Dict[str, List[Batch]], return list(batch.values()).
        """
        if isinstance(batch, list):
            return [batch]
        if isinstance(batch, dict):
            # CombinedLoader returns a dict of sequences (one per sub-dataset)
            return list(batch.values())
        raise TypeError(f"Unexpected batch type: {type(batch)}. Expected List[Batch] or Dict[str, List[Batch]].")

    def _prepare_graph(self, sequence: List[Batch]) -> Batch:
        graph = sequence_to_graph_single_target(sequence, use_past_as_cond=self.use_past_as_cond)
        # Lightning handles device placement for tensors that are already on device; make sure graph tensors follow
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
        if self.net.is_latent_diffusion:
            graph = self.net.autoencoder.transform(graph)

        # Sample diffusion steps and weights per-graph in batch
        batch_size = graph.batch.max().item() + 1
        r, sample_weight = self.step_sampler(batch_size, device)
        graph.r = r

        # Diffuse x0 -> xt
        graph.field_start = graph.x_latent_target if self.net.is_latent_diffusion else graph.target
        graph.field_r, graph.noise = self.dp(
            field_start    = graph.field_start,
            r              = graph.r,
            batch          = graph.batch,
            dirichlet_mask = None if self.net.is_latent_diffusion else getattr(graph, 'dirichlet_mask', None),
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
        # Compute loss per stream and average (you can weight if desired)
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
        # Log gradient L2 norm AFTER backward is called by Lightning (Option B).
        # With AMP enabled, these grads are scaled; it is still useful to monitor trends.
        if not self.training or (hasattr(self, 'trainer') and self.trainer.sanity_checking):
            return
        total = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total += float(g.norm(2).item() ** 2)
        grad_norm = total ** 0.5
        self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=False)