import os
from Dataset import create_cfd_datamodule,estimate_mesh_scales,_sample_graphs_for_stats
from utils import read_metadata,load_coo_data,create_data_list,create_graph_data
import lightning as L
from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.dgn_model import DiffusionGraphNet
from dgn4avbp.step_sampler import ImportanceStepSampler
from dgn4avbp.lit_dgn import LitDiffusionCFD
from dgn4avbp.losses import HybridLoss
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from torchvision import transforms as T
from torch_geometric.data import Data, Batch
import torch

from dgn4avbp.transform_locals import EnsureEdgeAttrFromPos, ScaleEdgeAttr, MeshCoarsening, ZScoreTarget,ScaleAttr
from torchvision import transforms as T
import torch

def _sample_graphs_for_stats(meta, K=4):
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
        case, sim, t = it_list_total[k][0]        # take the first step of each sequence
        fpath = file_list[str(case)][str(t)]
        g = create_graph_data(pos0, edge_index0, fpath, meta, cells0)
        samples.append(g)
    return samples, pos0, edge_index0

def _estimate_mesh_scales(pos, edge_index, quantile=0.5):
    row, col = edge_index
    edge_len = (pos[col] - pos[row]).norm(dim=1)
    h = edge_len.quantile(quantile).item()
    rel_pos_scaling = [h, 2*h, 4*h, 8*h]
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

        # will be set in setup()
        self.graph_transform = None
        self._zscore_mean = None
        self._zscore_std  = None
        self._rel_pos_scales = None
        self._h = None

    def setup(self, stage: str):
        # --- compute stats ONCE from the first dataset (or loop over all and concat) ---
        meta0 = read_metadata(self.metadata_files[0])
        samples, pos0, edge_index0 = _sample_graphs_for_stats(meta0, K=4)

        # target mean/std (your target has 3 velocity components)
        tgt = torch.cat([g.target for g in samples], dim=0)  # shape (N_total, 3)
        self._zscore_mean = tgt.mean(0)                      # (3,)
        self._zscore_std  = tgt.std(0)                       # (3,)
        print(self._zscore_mean,self._zscore_std)
        # mesh scales for edge normalization & LR scaling
        self._h, self._rel_pos_scales = _estimate_mesh_scales(pos0, edge_index0)

        # --- build the graph transform pipe (reused for train/val/test) ---
        self.graph_transform = T.Compose([
            EnsureEdgeAttrFromPos(),         # edge_attr = pos[j]-pos[i]
            ScaleEdgeAttr(self._h),          # normalize HR edge vectors by median spacing
            ZScoreTarget(self._zscore_mean, self._zscore_std),  # normalize targets
            MeshCoarsening(
                num_scales=4,
                max_indegree=None,           # set later for unstructured meshes
                rel_pos_scaling=None,
                scalar_rel_pos=False,         # keep 3D vectors for LR edges
            ),
        ])

        # --- create loaders with this transform applied in the Collater ---
        if stage == "fit":
            self.train_cfd_datamodule = create_cfd_datamodule(
                self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                shuffle=True, split=self.split, flag='train',
                collater_transform=self.graph_transform
            )
            self.val_cfd_datamodule = create_cfd_datamodule(
                self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                shuffle=False, split=self.split, flag='val',
                collater_transform=self.graph_transform
            )

        if stage == "test":
            self.val_cfd_datamodule = create_cfd_datamodule(
                self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                shuffle=False, split=self.split, flag='val',
                collater_transform=self.graph_transform
            )
        if stage == "predict":
            self.val_cfd_datamodule = create_cfd_datamodule(
                self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                shuffle=False, split=self.split, flag='val',
                collater_transform=self.graph_transform
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
    




metadata_files = [
        os.path.join('/scratch/coop/theret/cfd-dataset/tutorial/sample_dataset/metadata.yaml')
   ]

graph_transform = T.Compose([
    EnsureEdgeAttrFromPos(),                # builds base edge_attr from pos (if you don’t already)
    ScaleEdgeAttr(0.015),                   # like DGN4CFD
    #EdgeCondFreeStreamLocalAxes([...]),     # optional: your edge_cond
    ScaleAttr('target', vmin=0, vmax=1000),# your ranges
    MeshCoarsening(
        num_scales=4,
        max_indegree=None,
        #rel_pos_scaling=[0.015, 0.03, 0.06, 0.12],
        scalar_rel_pos=True,
    ),
])

# Diffusion process
diffusion_process = DiffusionProcess(
    num_steps     = 1000,
    schedule_type = 'linear',
)

# Model
arch = {
    'in_node_features':   3,
    'cond_node_features': 0,
    'cond_edge_features': 3,
    'depths':             [3,3,3,3],
    'fnns_width':         128,
    'aggr':               'sum',
    'dropout':            0.1,
    'dim':                  3,
    "scalar_rel_pos":   False
}
net = DiffusionGraphNet(
    diffusion_process  = diffusion_process,
    learnable_variance = True,
    arch               = arch,
)

# Loss and sampler
criterion = HybridLoss()              # (model, graph) -> [B]
step_sampler_factory = ImportanceStepSampler

# LightningModule wrapper
lit = LitDiffusionCFD(
    net=net,
    diffusion_process=diffusion_process,
    criterion=criterion,
    step_sampler_factory=step_sampler_factory,
    lr=1e-4,
    scheduler_cfg={"factor":0.1, "patience":50},
    pack_mode="y_window_cond_static",
    pack_win_len=1,            # <— use these names
    pack_stride=1,
    pack_select="random",
    y_idx=[0,1,2],
    cond_idx=None,
)


# DataModule from your CFDDataset
dm = cfd_datamodule(metadata_files, train_val_split=0.8)
#dm.setup(stage='fit')
# Get the combined loader
# combined_loader = dm.train_dataloader()
# cfd_datamodule_train  = dm.train_cfd_datamodule
# print(f"Number of subdatasets: {len(cfd_datamodule_train.subdatasets)}")
# for i, subdataset in enumerate(cfd_datamodule_train.subdatasets):
#         print(f"Subdataset {i+1} batch size: {dm.batch_sizes[i]}")

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
    
# Trainer
prog_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow", 
                                                      progress_bar="green1", 
                                                      progress_bar_finished="green1", 
                                                      progress_bar_pulse="#6206E0", 
                                                      batch_progress="green_yellow", 
                                                      time="grey82", 
                                                      processing_speed="grey82", 
                                                      metrics="grey82", 
                                                      metrics_text_delimiter="\n", 
                                                      metrics_format=".3e"))

ckpt = ModelCheckpoint(dirpath="checkpoints", 
                       filename="diffusion-{epoch}", 
                       monitor="val/loss", 
                       mode="min", 
                       save_top_k=3,
                       save_last=True)

trainer = L.Trainer(max_epochs=30, 
                    accelerator="auto", 
                    precision="16-mixed", 
                    callbacks=[ckpt, prog_bar], 
                    log_every_n_steps=10, 
                    limit_val_batches=20, 
                    limit_train_batches=80,
                    accumulate_grad_batches=64)

# Train
# after you construct `trainer`, `lit`, and `dm`:
ckpt_path = "/scratch/coop/theret/nn4avbp/checkpoints/last.ckpt"  # or a specific epoch file like "checkpoints/diffusion-epoch=1.ckpt"
if ckpt_path is None:
    trainer.fit(lit, dm) 
else:
    trainer.fit(lit, dm, ckpt_path=ckpt_path)

# # 1) Rebuild lit exactly as for training
# lit = LitDiffusionCFD(
#     net=net,
#     diffusion_process=diffusion_process,
#     criterion=criterion,
#     step_sampler_factory=step_sampler_factory,
#     lr=1e-4,
#     scheduler_cfg={"factor":0.1, "patience":50},
#     pack_mode="y_window_cond_static",
#     pack_win_len=1,
#     pack_stride=1,
#     pack_select="random",
#     y_idx=[1,2,3],
#     cond_idx=None,
# )

# # 2) Load state dict directly (skip Lightning’s legacy patcher)
# ckpt = torch.load("checkpoints/last.ckpt", map_location="cpu")
# lit.load_state_dict(ckpt["state_dict"], strict=True)  # strict=False if you changed the model
# lit.eval().freeze()



# # 3) Predict as usual

# trainer = L.Trainer( 
#                     accelerator="auto", 
#                     precision="16-mixed", 
#                     callbacks=[ckpt], 
#                     log_every_n_steps=10, 
#                     limit_val_batches=20, 
#                     limit_train_batches=80,
#                     accumulate_grad_batches=64)


# dm.setup(stage="predict")
# pred = trainer.predict(lit, dm)

