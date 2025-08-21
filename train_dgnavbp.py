import os
from Dataset import create_cfd_datamodule
from utils import read_metadata
import lightning as L
from dgn4avbp.diffusion_process import DiffusionProcess
from dgn4avbp.dgn_model import DiffusionGraphNet
from dgn4avbp.step_sampler import ImportanceStepSampler
from dgn4avbp.lit_dgn import LitDiffusionCFD
from dgn4avbp.losses import HybridLoss
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

class cfd_datamodule(L.LightningDataModule):
    def __init__(self, metadata_files, train_val_split = 0.8):
        super().__init__()
        
        self.metadata_files = metadata_files
        metadata =  [read_metadata(metadata_file) for metadata_file in metadata_files]
        # Specify batch sizes for each dataset
        self.batch_sizes = [info['batch_size'] for info in metadata]  # Adjust these as needed
        print(self.batch_sizes)
        # Specify loader types for each dataset
        self.loader_types = ['default']*len(metadata_files)  # Example: using different loader types
    
        # Specify at what index we start sampling from Dataset
        self.start_idx = [0]*len(metadata_files)
        self.split = train_val_split

    def prepare_data(self):
        None

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_cfd_datamodule = create_cfd_datamodule(self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                                        shuffle=True, split=self.split, flag='train')
            self.val_cfd_datamodule = create_cfd_datamodule(self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                                        shuffle=False, split=self.split, flag='val')
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.val_cfd_datamodule = create_cfd_datamodule(self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                                        shuffle=False, split=self.split, flag='val')
        if stage == "predict":
            self.val_cfd_datamodule = create_cfd_datamodule(self.metadata_files, self.batch_sizes, self.loader_types, self.start_idx,
                                        shuffle=False, split=self.split, flag='val')

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

# Diffusion process
diffusion_process = DiffusionProcess(
    num_steps     = 1000,
    schedule_type = 'linear',
)

# Model
arch = {
    'in_node_features':   6,
    'cond_node_features': 2,
    'cond_edge_features': 3,
    'depths':             [3],
    'fnns_width':         128,
    'aggr':               'sum',
    'dropout':            0.1,
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
    y_idx=[0],
    cond_idx=[1,2,3,4,5,6,7],
)


# DataModule from your CFDDataset
dm = cfd_datamodule(metadata_files, train_val_split=0.8)
dm.setup(stage='fit')
# Get the combined loader
combined_loader = dm.train_dataloader()
cfd_datamodule_train  = dm.train_cfd_datamodule
print(f"Number of subdatasets: {len(cfd_datamodule_train.subdatasets)}")
for i, subdataset in enumerate(cfd_datamodule_train.subdatasets):
        print(f"Subdataset {i+1} batch size: {dm.batch_sizes[i]}")

from torch_geometric.data import Data, Batch

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
prog_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow", progress_bar="green1", progress_bar_finished="green1", progress_bar_pulse="#6206E0", batch_progress="green_yellow", time="grey82", processing_speed="grey82", metrics="grey82", metrics_text_delimiter="\n", metrics_format=".3e"))
ckpt = ModelCheckpoint(dirpath="checkpoints", filename="diffusion-{epoch}", monitor="val/loss", mode="min", save_top_k=3)
trainer = L.Trainer(max_epochs=2, 
                    accelerator="auto", 
                    precision="16-mixed", 
                    callbacks=[ckpt, prog_bar], 
                    log_every_n_steps=10, 
                    limit_val_batches=20, 
                    limit_train_batches=80,
                    accumulate_grad_batches=64)

# Train
trainer.fit(lit, dm) 

