import argparse
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBarTheme,
)
from torch_geometric.data import Batch, Data
from torchvision import transforms as T

from Dataset import (
    create_cfd_datamodule,
    estimate_mesh_scales,
    _sample_graphs_for_stats,
)
from utils import read_metadata

from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.dgn_model import DiffusionGraphNet
from dgn4avbp.lit_dgn import LitDiffusionCFD
from dgn4avbp.losses import HybridLoss
from dgn4avbp.step_sampler import ImportanceStepSampler
from dgn4avbp.transform_locals import (
    EnsureEdgeAttrFromPos,
    MeshCoarsening,
    ScaleEdgeAttr,
    ZScoreTarget,
)

class cfd_datamodule(L.LightningDataModule):
    def __init__(self, metadata_files, train_val_split=0.8):
        super().__init__()
        self.metadata_files = metadata_files
        self._metadata = [read_metadata(mf) for mf in metadata_files]
        self.batch_sizes = [info["batch_size"] for info in self._metadata]
        self.loader_types = ["default"] * len(metadata_files)
        self.start_idx = [0] * len(metadata_files)
        self.split = train_val_split

        # will be set in setup()
        self.graph_transform = None
        self._zscore_mean = None
        self._zscore_std  = None
        self._rel_pos_scales = None
        self._h = None

    def setup(self, stage: str | None = None):
        if self.graph_transform is None:
            meta0 = self._metadata[0]
            samples, pos0, edge_index0 = _sample_graphs_for_stats(meta0, K=4)

            tgt = torch.cat([g.target for g in samples], dim=0)
            self._zscore_mean = tgt.mean(0)
            self._zscore_std = tgt.std(0)

            self._h, self._rel_pos_scales = estimate_mesh_scales(pos0, edge_index0)

            self.graph_transform = T.Compose(
                [
                    EnsureEdgeAttrFromPos(),
                    ScaleEdgeAttr(self._h),
                    ZScoreTarget(self._zscore_mean, self._zscore_std),
                    MeshCoarsening(
                        num_scales=4,
                        max_indegree=None,
                        rel_pos_scaling=self._rel_pos_scales,
                        scalar_rel_pos=False,
                    ),
                ]
            )

        if stage in (None, "fit"):
            self.train_cfd_datamodule = create_cfd_datamodule(
                self.metadata_files,
                self.batch_sizes,
                self.loader_types,
                self.start_idx,
                shuffle=True,
                split=self.split,
                flag="train",
                collater_transform=self.graph_transform,
            )

        if stage in (None, "fit", "validate", "test", "predict"):
            self.val_cfd_datamodule = create_cfd_datamodule(
                self.metadata_files,
                self.batch_sizes,
                self.loader_types,
                self.start_idx,
                shuffle=False,
                split=self.split,
                flag="val",
                collater_transform=self.graph_transform,
            )

    def prepare_data(self):
        None

    def train_dataloader(self):
        return self.train_cfd_datamodule.get_combined_loader()

    def val_dataloader(self):
        return self.val_cfd_datamodule.get_combined_loader(mode='sequential')

    def test_dataloader(self):
        return self.val_cfd_datamodule.get_combined_loader(mode='sequential')

    def predict_dataloader(self):
        return self.val_cfd_datamodule.get_combined_loader(mode='sequential')
    
DEFAULT_METADATA = "/scratch/coop/theret/cfd-dataset/tutorial/sample_dataset/metadata.yaml"


def build_lit_module() -> LitDiffusionCFD:
    diffusion_process = DiffusionProcess(
        num_steps=1000,
        schedule_type="linear",
    )

    arch = {
        "in_node_features": 3,
        "cond_node_features": 0,
        "cond_edge_features": 3,
        "depths": [2],
        "fnns_width": 128,
        "aggr": "sum",
        "dropout": 0.1,
        "dim": 3,
        "scalar_rel_pos": False,
    }

    net = DiffusionGraphNet(
        diffusion_process=diffusion_process,
        learnable_variance=True,
        arch=arch,
    )

    criterion = HybridLoss()
    step_sampler_factory = ImportanceStepSampler

    return LitDiffusionCFD(
        net=net,
        diffusion_process=diffusion_process,
        criterion=criterion,
        step_sampler_factory=step_sampler_factory,
        lr=1e-4,
        scheduler_cfg={"factor": 0.1, "patience": 50},
        pack_mode="y_window_cond_static",
        pack_win_len=1,
        pack_stride=1,
        pack_select="random",
        y_idx=[0, 1, 2],
        cond_idx=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DGN4AVBP diffusion model")
    parser.add_argument(
        "--metadata",
        nargs="+",
        default=None,
        help="Path(s) to metadata YAML files. Defaults to CFD_METADATA env var or sample dataset.",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.8,
        help="Fraction of sequences to use for training.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory where Lightning checkpoints are stored.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=80,
        help=(
            "Number of train batches per epoch (Lightning Trainer setting). "
            "Defaults to 80 to preserve the prior quick-debug configuration."
        ),
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=20,
        help=(
            "Number of validation batches per epoch (Lightning Trainer setting). "
            "Defaults to 20 to match the historical script behavior."
        ),
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=64,
        help="Gradient accumulation factor.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="Logging frequency for the Trainer.",
    )
    return parser.parse_args()


def resolve_metadata_paths(metadata_args: list[str] | None) -> list[str]:
    if metadata_args:
        return metadata_args
    env_metadata = os.getenv("CFD_METADATA")
    if env_metadata:
        return [path for path in env_metadata.split(os.pathsep) if path]
    return [DEFAULT_METADATA]


def build_trainer(args: argparse.Namespace) -> L.Trainer:
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    prog_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )

    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="diffusion-noMuGNN-{epoch}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    trainer_kwargs: dict[str, object] = dict(
        max_epochs=args.max_epochs,
        accelerator="auto",
        precision="16-mixed",
        callbacks=[checkpoint, prog_bar],
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    return L.Trainer(**trainer_kwargs)

def get_sequence_from_combined(item):
    # item may be: (batches, batch_idx, dataloader_idx) OR just `batches`
    if isinstance(item, tuple) and len(item) == 3:
        batches, _batch_idx, _dl_idx = item
    else:
        batches = item

    # If CombinedLoader was built as {"main": loader}
    if isinstance(batches, dict):
        sequence = next(iter(batches.values()))
    else:
        sequence = batches

    # Some wrappers yield a tuple rather than list — treat them the same.
    if isinstance(sequence, tuple):
        sequence = list(sequence)
    return sequence  # list[Batch]

def print_batch_info(sequence):
    assert isinstance(sequence, (list, tuple)), f"Expected list/tuple, got {type(sequence)}"
    print(f"    Number of time steps in batch: {len(sequence)}")
    for t, g in enumerate(sequence):
        # If someone tucked the Batch inside {"main": Batch} per time-step:
        if isinstance(g, dict) and "main" in g:
            g = g["main"]
        assert isinstance(g, (Batch, Data)), f"Time step {t} is {type(g)}"
        print(f"    Time step {t+1}:")
        print(f"        Number of graphs in batch: {g.num_graphs}")
        print(f"        Total number of nodes: {g.num_nodes}")
        print(f"        Total number of edges: {g.num_edges}")
        if hasattr(g, 'target'):
            print(f"        target shape: {tuple(g.target.shape)}")
        if hasattr(g, 'edge_attr'):
            print(f"        edge_attr shape: {tuple(g.edge_attr.shape)}")
        if hasattr(g, 'cells') and isinstance(g.cells, list) and g.cells and g.cells[0] is not None:
            print(f"        Total number of cells: {len(g.cells[0])}")
        if hasattr(g, 'time'):
            print(f"        time mean per-graph: {g.time.view(g.num_graphs, -1).mean(dim=-1)}")

# # Iterate safely
# for i, item in enumerate(combined_loader):
#     print(f"\nBatch {i+1}:")
#     sequence = get_sequence_from_combined(item)
#     print("Dataloader main (Type: Default):")
#     print_batch_info(sequence)
    
def main() -> None:
    args = parse_args()
    metadata_files = resolve_metadata_paths(args.metadata)

    lit_module = build_lit_module()
    dm = cfd_datamodule(metadata_files, train_val_split=args.train_val_split)
    trainer = build_trainer(args)

    ckpt_path = args.resume_from if args.resume_from else None
    trainer.fit(lit_module, dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

