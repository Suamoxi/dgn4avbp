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
    'in_node_features':   8,
    'cond_node_features': 0,
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
    use_past_as_cond=False,    # set True if you want to concatenate past frames into .cond
)

# DataModule from your CFDDataset
dm = cfd_datamodule(metadata_files, train_val_split=0.8)

# Trainer
prog_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow", progress_bar="green1", progress_bar_finished="green1", progress_bar_pulse="#6206E0", batch_progress="green_yellow", time="grey82", processing_speed="grey82", metrics="grey82", metrics_text_delimiter="\n", metrics_format=".3e"))
ckpt = ModelCheckpoint(dirpath="checkpoints", filename="diffusion-{epoch}", monitor="val/loss", mode="min", save_top_k=3)
trainer = L.Trainer(max_epochs=2, accelerator="auto", precision="16-mixed", callbacks=[ckpt, prog_bar], log_every_n_steps=10, limit_val_batches=20, limit_train_batches=80)

# Train
trainer.fit(lit, dm)