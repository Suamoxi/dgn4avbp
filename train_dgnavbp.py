#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from datetime import datetime

import torch
import lightning as L
from torch_geometric.data import Data, Batch
from torchvision import transforms as T

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks import Callback

# --- your domain imports ---
from Dataset import create_cfd_datamodule, estimate_mesh_scales, _sample_graphs_for_stats
from utils import read_metadata, load_coo_data, create_data_list, create_graph_data
from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.dgn_model import DiffusionGraphNet
from dgn4avbp.step_sampler import ImportanceStepSampler
from dgn4avbp.lit_dgn import LitDiffusionCFD
from dgn4avbp.losses import HybridLoss
from dgn4avbp.transform_locals import (
    EnsureEdgeAttrFromPos, ScaleEdgeAttr, MeshCoarsening, ZScoreTarget, ScaleAttr
)
from dgn4avbp.callbacks import StepTimeTracker


# =========================
# Logging & utility helpers
# =========================
os.environ.setdefault("PYTHONUNBUFFERED", "1")

LOG_DIR = os.environ.get("LOG_DIR", f"/scratch/{os.environ.get('USER','user')}/logs")
os.makedirs(LOG_DIR, exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
file_log_path = os.path.join(LOG_DIR, f"train_{_ts}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),   # -> captured by Slurm .out
        logging.FileHandler(file_log_path)   # -> persistent text log
    ]
)
log = logging.getLogger("train")
log.info("Initialized logging. Slurm .out and file logs are active.")
log.info(f"LOG_DIR={LOG_DIR}")

def get_sequence_from_combined(item):
    if isinstance(item, tuple) and len(item) == 3:
        batches, _batch_idx, _dl_idx = item
    else:
        batches = item
    if isinstance(batches, dict):
        sequence = next(iter(batches.values()))
    else:
        sequence = batches
    if isinstance(sequence, tuple):
        sequence = list(sequence)
    return sequence  # list[Batch]


# =========================
# Datamodule (yours, + logs)
# =========================
def _sample_graphs_for_stats(meta, K=200):
    pos0, edge_index0, cells0 = load_coo_data(
        meta['coo_file'], meta['mesh_file'], meta['coordinate_paths']
    )
    file_list, it_list_total = create_data_list(
        [meta['solution_directory']],
        meta['seq_len'],
        meta['solution_prefix']
    )
    K = min(K, len(it_list_total))
    samples = []
    for k in range(K):
        case, sim, t = it_list_total[k][0]        # first step of each sequence
        fpath = file_list[str(case)][str(t)]
        g = create_graph_data(pos0, edge_index0, fpath, meta, cells0)
        samples.append(g)
    return samples, pos0, edge_index0

def _estimate_mesh_scales(pos, edge_index, quantile=0.5):
    row, col = edge_index
    edge_len = (pos[col] - pos[row]).norm(dim=1)
    h = edge_len.quantile(quantile).item()
    rel_pos_scaling = [h, 2 * h, 4 * h, 8 * h]
    return h, rel_pos_scaling

class cfd_datamodule(L.LightningDataModule):
    def __init__(self, metadata_files, train_val_split=0.8):
        super().__init__()
        self.metadata_files = metadata_files
        metadata = [read_metadata(mf) for mf in metadata_files]
        self.batch_sizes  = [info['batch_size'] for info in metadata]
        self.loader_types = ['default'] * len(metadata_files)
        self.start_idx    = [0] * len(metadata_files)
        self.split        = train_val_split

        self.graph_transform = None
        self._zscore_mean = None
        self._zscore_std  = None
        self._rel_pos_scales = None
        self._h = None

    def setup(self, stage: str):
        meta0 = read_metadata(self.metadata_files[0])
        samples, pos0, edge_index0 = _sample_graphs_for_stats(meta0, K=200)

        tgt = torch.cat([g.target for g in samples], dim=0)  # (N_total, 3)
        self._zscore_mean = tgt.mean(0)
        self._zscore_std  = tgt.std(0)
        log.info(f"zscore_mean={self._zscore_mean.tolist()} zscore_std={self._zscore_std.tolist()}")

        self._h, self._rel_pos_scales = _estimate_mesh_scales(pos0, edge_index0)
        log.info(f"median_h={self._h:.6e} rel_pos_scales={self._rel_pos_scales}")

        self.graph_transform = T.Compose([
            EnsureEdgeAttrFromPos(),
            ScaleEdgeAttr(1/self._h),
            ZScoreTarget(self._zscore_mean, self._zscore_std),
            MeshCoarsening(
                num_scales=5,
                max_indegree=None,
                rel_pos_scaling=None,
                scalar_rel_pos=False,
            ),
        ])

        cpu_count = os.cpu_count() or 1
        # Validation tends to benefit from more workers because it cannot hide
        # latency behind backprop; train uses roughly half to leave room for the
        # training loop itself.
        train_workers = max(1, min(8, cpu_count // 2 or 1))
        val_workers = max(1, min(8, cpu_count))

        def _loader_kwargs(num_workers: int, *, prefetch_factor: int) -> dict:
            # Centralised helper so both train/val loaders share the same logic
            # for enabling worker reuse and pinned memory when available.
            kwargs = {
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
            }
            if num_workers > 0:
                kwargs["persistent_workers"] = True
            if torch.cuda.is_available():
                kwargs["pin_memory"] = True
            return kwargs

        train_loader_kwargs = _loader_kwargs(train_workers, prefetch_factor=2)
        val_loader_kwargs = _loader_kwargs(val_workers, prefetch_factor=4 if val_workers > 1 else 2)

        # Emit the final settings in the log so we can correlate them with
        # measured data_step_time changes when running experiments.
        log.info(f"Train DataLoader kwargs: {train_loader_kwargs}")
        log.info(f"Val DataLoader kwargs: {val_loader_kwargs}")

        if stage == "fit":
            self.train_cfd_datamodule = create_cfd_datamodule(
                self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                shuffle=True, split=self.split, flag='train',
                collater_transform=self.graph_transform,
                dataloader_kwargs=train_loader_kwargs,
                cache_sequences=False,
            )
            self.val_cfd_datamodule = create_cfd_datamodule(
                self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                shuffle=False, split=self.split, flag='val',
                collater_transform=self.graph_transform,
                dataloader_kwargs=val_loader_kwargs,
                # Validation/test reuse a deterministic index order. Enable the
                # dataset-level cache so we only pay the HDF5 + SymPy cost once
                # per sequence and serve clones on subsequent epochs.
                cache_sequences=True,
            )
        if stage in {"test", "predict"}:
            self.val_cfd_datamodule = create_cfd_datamodule(
                self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                shuffle=False, split=self.split, flag='val',
                collater_transform=self.graph_transform,
                dataloader_kwargs=val_loader_kwargs,
                cache_sequences=True,
            )

    def train_dataloader(self):
        return self.train_cfd_datamodule.get_combined_loader()

    def val_dataloader(self):
        return self.val_cfd_datamodule.get_combined_loader(mode='sequential')

    def test_dataloader(self):
        return self.val_cfd_datamodule.get_combined_loader(mode='sequential')

    def predict_dataloader(self):
        return self.val_cfd_datamodule.get_combined_loader(mode='sequential')


# =========================
# Model & training objects
# =========================
metadata_files = [os.path.join('/scratch/coop/theret/HIT_LES_FORCED/metadata.yaml')]

diffusion_process = DiffusionProcess(
    num_steps     = 1000,
    schedule_type = 'linear',
)

arch = {
    'in_node_features':   3,
    'cond_node_features': 0,
    'cond_edge_features': 3,
    'depths':             [3,3,3,3,3],
    'fnns_width':         128,
    'aggr':               'sum',
    'dropout':            0.1,
    'dim':                3,
    "scalar_rel_pos":     False
}
net = DiffusionGraphNet(
    diffusion_process  = diffusion_process,
    learnable_variance = True,
    arch               = arch,
)

criterion = HybridLoss()
step_sampler_factory = ImportanceStepSampler

lit = LitDiffusionCFD(
    net=net,
    diffusion_process=diffusion_process,
    criterion=criterion,
    step_sampler_factory=step_sampler_factory,
    lr=1e-4,
    scheduler_cfg={"factor":0.1, "patience":250},
    pack_mode=None,
    pack_win_len=10,
    pack_stride=1,
    pack_select="random",
    y_idx=[0,1,2],
    cond_idx=None,
)

dm = cfd_datamodule(metadata_files, train_val_split=0.8)
# dm.setup(stage='fit')

# # quick inspection (logged)
# dl = dm.train_dataloader()
# item = next(iter(dl))
# sequence = get_sequence_from_combined(item)
# g = sequence[-1]
# log.info(f"seq_len={len(sequence)} num_graphs={g.num_graphs} num_nodes={g.num_nodes} "
#          f"target_shape={tuple(g.target.shape)} "
#          f"edge_attr_shape={tuple(g.edge_attr.shape) if hasattr(g,'edge_attr') else None}")


# =========================
# Callbacks: progress, LR, GPU
# =========================
USE_RICH = sys.stdout.isatty() and os.environ.get("LIGHTNING_DISABLE_PROGRESS_BAR","0") != "1"
prog_bar = None
if USE_RICH:
    prog_bar = RichProgressBar(theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
        metrics_text_delimiter="\n",
        metrics_format=".3e"
    ))

class CompactConsoleLogger(Callback):
    """Epoch-level metric/LR printer to stdout (Slurm .out)."""
    def _fmt(self, x):
        try:
            return f"{float(x):.6e}"
        except Exception:
            return str(x)

    def _current_lrs(self, trainer):
        lrs = []
        try:
            for opt in trainer.optimizers:
                for pg in opt.param_groups:
                    lr = pg.get("lr", None)
                    if lr is not None:
                        lrs.append(lr)
        except Exception:
            pass
        return lrs

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        lrs = self._current_lrs(trainer)
        lr_str = ",".join(self._fmt(lr) for lr in lrs) if lrs else "nan"
        msg = (
            f"epoch={trainer.current_epoch} "
            f"train_loss_epoch={self._fmt(m.get('train_loss_epoch','nan'))} "
            f"val_loss={self._fmt(m.get('val/loss','nan'))} "
            f"lr={lr_str}"
        )
        log.info(msg)

class GPUMonitor(Callback):
    """Logs GPU memory and optional utilization every N steps."""
    def __init__(self, log_every_n_steps=50):
        self.every = int(log_every_n_steps)

    def _maybe_query_nvidia_smi(self):
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            ).stdout.strip()
            # If multiple GPUs, show only the one Lightning is likely using (index 0)
            return out.splitlines()[0] if out else None
        except Exception:
            return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.every != 0:
            return
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024**2)
            mem_reserved = torch.cuda.memory_reserved() / (1024**2)
            max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            msg = (f"GPU mem (MB): alloc={mem_alloc:.1f} reserved={mem_reserved:.1f} "
                   f"max_alloc={max_alloc:.1f}")
            util = self._maybe_query_nvidia_smi()
            if util:
                msg += f" | nvidia-smi (util%, tempC): {util}"
            log.info(msg)

class StepConsoleLogger(Callback):
    def __init__(self, every=50, warmup=5):
        self.every = int(every)
        self.warmup = int(warmup)
    def _fmt(self, x):
        try:
            return f"{float(x):.6e}"
        except Exception:
            return str(x)
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx + 1 < self.warmup:
            return
        if (batch_idx + 1) % self.every != 0:
            return
        loss_val = None
        if isinstance(outputs, torch.Tensor):
            loss_val = outputs.detach()
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss_val = outputs["loss"]
        if loss_val is None:
            loss_val = trainer.callback_metrics.get("train/loss") or trainer.callback_metrics.get("train_loss")
        lrs = []
        try:
            for opt in trainer.optimizers:
                for pg in opt.param_groups:
                    lr = pg.get("lr", None)
                    if lr is not None:
                        lrs.append(lr)
        except Exception:
            pass
        lr_str = ",".join(self._fmt(lr) for lr in lrs) if lrs else "nan"
        msg = f"step={batch_idx+1} loss={self._fmt(loss_val) if loss_val is not None else 'nan'} lr={lr_str}"
        log.info(msg)



# =========================
# Checkpointing & Trainer
# =========================
ckpt_dir = os.path.join(LOG_DIR, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename="diffusion-MuGNN-lowresHIT-{epoch}",
    monitor="val/loss",
    mode="min",
    save_top_k=3,
    save_last=True
)

csv_logger = CSVLogger(save_dir=LOG_DIR, name="lightning_csv")
tb_logger  = TensorBoardLogger(save_dir=LOG_DIR, name="lightning_tb")

time_tracker = StepTimeTracker(warmup_batches=0, log_every_n_steps=1)

_callbacks = [ckpt, time_tracker, CompactConsoleLogger(), GPUMonitor(log_every_n_steps=1),StepConsoleLogger(every=10,warmup=0)]
if prog_bar is not None:
    _callbacks.insert(0, prog_bar)

trainer = L.Trainer(
    max_epochs=5000,
    accelerator="auto",
    precision="16-mixed",
    callbacks=_callbacks,
    logger=[csv_logger, tb_logger],
    enable_progress_bar=(prog_bar is not None),
    log_every_n_steps=1,
    limit_val_batches=40,
    limit_train_batches=160,
    accumulate_grad_batches=4,
    gradient_clip_val=1.0,
    default_root_dir=LOG_DIR
)

# =========================
# (Optional) resume
# =========================
#ckpt_path = "/scratch/coop/theret/nn4avbp/checkpoints/last-v6.ckpt"
ckpt_path = None
if ckpt_path and not os.path.isabs(ckpt_path):
    ckpt_path = os.path.abspath(ckpt_path)
if ckpt_path and os.path.exists(ckpt_path):
    log.info(f"Resuming from checkpoint: {ckpt_path}")
else:
    if ckpt_path:
        log.warning(f"Checkpoint not found: {ckpt_path} (starting from scratch)")
        ckpt_path = None

# =========================
# GO
# =========================
if ckpt_path is None:
    trainer.fit(lit, dm)
else:
    trainer.fit(lit, dm, ckpt_path=ckpt_path)
