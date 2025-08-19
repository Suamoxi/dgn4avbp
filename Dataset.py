from torch.utils.data import Dataset,DataLoader,IterableDataset
from dgn4avbp.loader import Collater
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NodeLoader, NeighborLoader
from torch_geometric.utils import to_undirected
from utils import read_metadata, load_coo_data, load_simulation_data, create_graph_data, create_data_list
from lightning.pytorch.utilities import CombinedLoader
from torchvision import transforms as tv_transforms
import torch

class CFDSubDataset(IterableDataset):
    def __init__(self, metadata_file, start_idx, shuffle=False, split=1.0, flag: str = 'train'):
        self.metadata = read_metadata(metadata_file)
        self.node_positions, self.edge_indices, self.cells = load_coo_data(
            self.metadata['coo_file'], 
            self.metadata['mesh_file'], 
            self.metadata['coordinate_paths']
        )
        self.edge_indices = to_undirected(self.edge_indices)
        self.file_list, self.it_list_total = create_data_list(
            [self.metadata['solution_directory']],
            self.metadata['seq_len'],
            self.metadata['solution_prefix']
        )
        self.start_idx =  start_idx
        self.end_idx = len(self.it_list_total)
        self.split = split
        self.flag = flag
        self.shuffle = shuffle
        if flag == 'train':
            self.idx = torch.arange(self.start_idx, int(split*self.end_idx))
        else:
            self.idx = torch.arange(int(split*self.end_idx), self.end_idx)
        if self.shuffle:
            self.current_idx = torch.randperm(len(self.idx))
        else:
            self.current_idx = torch.arange(len(self.idx))
        self.count = 0

    def getitem(self, idx):
        sequence = []
        for i in range(self.metadata['seq_len']):
            case, sim, time_step = self.it_list_total[idx][i]
            file_name = self.file_list[str(case)][str(time_step)]
            graph_data = create_graph_data(self.node_positions, self.edge_indices, file_name, self.metadata, self.cells)
            sequence.append(graph_data)
        return sequence
    
    def __iter__(self):
        while True:
            yield self.getitem(self.idx[self.current_idx[self.count]])  # Generate data using the function
            self.count += 1

            # Reset the index if it reaches the end
            if self.count == len(self.idx):
                self.count = 0
                if self.shuffle:
                    self.current_idx = torch.randperm(len(self.idx))

class CFDDataset:
    def __init__(self, 
                 metadata_files, 
                 batch_sizes, 
                 loader_types, 
                 start_idx, 
                 shuffle=False, 
                 split=1.0, 
                 flag='train', 
                 nodes_per_sample=None,
                 collater_transform: tv_transforms.Compose = None,  # <- optional, applied on the BATCH
    ):
        self.subdatasets = []
        self.dataloaders = []
        self.collater = Collater(transform=collater_transform)

        for metadata_file, batch_size, loader_type, start in zip(metadata_files, batch_sizes, loader_types, start_idx):
            subdataset = CFDSubDataset(metadata_file, start, shuffle, split, flag)
            self.subdatasets.append(subdataset)
        
        for subdataset in self.subdatasets:
            dataloader = self.create_dataloader(subdataset, batch_size, loader_type, nodes_per_sample)
            self.dataloaders.append(dataloader)
    
    def _sequence_collate(self, batch: list[list[Data]]):
        """batch is a list of sequences; each sequence is a list[Data] of length seq_len.
           We collate per time-step using your Collater (which fixes multiscale indices)."""
        seq_len = len(batch[0])
        batched_sequence = []
        for t in range(seq_len):
            graphs_at_t = [seq[t] for seq in batch]                       # List[Data] at time t
            batched_graphs = self.collater.collate(graphs_at_t)           # <- use your Collater here
            batched_sequence.append(batched_graphs)                       # Batch
        return batched_sequence                                           # List[Batch] length seq_len
    
    def create_dataloader(self, subdataset, batch_size, loader_type, nodes_per_sample):
        common_kwargs = {
            'batch_size': batch_size,
            'collate_fn': self.collate_fn,
            'num_workers': 0
        }

        if loader_type == 'default':
            return DataLoader(subdataset, **common_kwargs)
        elif loader_type == 'node':
            return NodeLoader(
                subdataset,
                num_nodes=nodes_per_sample,
                **common_kwargs
            )
        elif loader_type == 'neighbor':
            return NeighborLoader(
                subdataset,
                num_neighbors=[8, 8, 8],  # Adjust these values as needed
                **common_kwargs
            )
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")

    @staticmethod
    def collate_fn(batch):
        # This function will batch graphs spatially for each time step
        seq_len = len(batch[0])
        batched_sequence = []
        for t in range(seq_len):
            graphs_at_t = [seq[t] for seq in batch]
            batched_graphs = Batch.from_data_list(graphs_at_t)
            batched_sequence.append(batched_graphs)
        return batched_sequence

    def get_combined_loader(self, mode="max_size_cycle"):
        if len(self.dataloaders) == 1:
            return CombinedLoader({"main": self.dataloaders[0]}, mode=mode)
        return CombinedLoader(self.dataloaders, mode=mode)

def create_cfd_datamodule(metadata_files, batch_sizes, loader_types, start_idx, shuffle=False, split=1.0, flag='train', nodes_per_sample=None):
    return CFDDataset(metadata_files, batch_sizes, loader_types, start_idx, shuffle, split, flag, nodes_per_sample)