from torch.utils.data import Dataset,IterableDataset
from dgn4avbp.loader import Collater
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NodeLoader, NeighborLoader
from torch_geometric.utils import to_undirected, coalesce
from utils import read_metadata, load_coo_data, load_simulation_data, create_graph_data, create_data_list
from lightning.pytorch.utilities import CombinedLoader
from torchvision import transforms as tv_transforms
import torch
from torchvision import transforms as T

from torch.utils.data import DataLoader as TorchDataLoader

class CFDSubDataset(IterableDataset):
    def __init__(self, metadata_file, start_idx, shuffle=False, split=1.0, flag: str = 'train'):
        self.metadata = read_metadata(metadata_file)
        self.node_positions, self.edge_indices, self.cells = load_coo_data(
            self.metadata['coo_file'], 
            self.metadata['mesh_file'], 
            self.metadata['coordinate_paths']
        )
        self.edge_indices = to_undirected(self.edge_indices)
        self.edge_indices = coalesce(self.edge_indices, sort_by_row=False)
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
        metadata_files, batch_sizes, loader_types, start_idx, 
        shuffle=False, split=1.0, flag='train', nodes_per_sample=None,
        collater_transform=None,   # <— add this
        dataloader_kwargs=None,
    ):
        self.subdatasets = []
        self.dataloaders = []
        # Keep a single Collater instance so that transforms (e.g. mesh
        # coarsening) are reused across every time step we collate.
        self.collater = Collater(transform=collater_transform)  # <— use it
        # Extra keyword arguments allow the caller to tune worker/prefetch
        # behaviour per-stage (train/val/test) without rewriting this class.
        self._dataloader_kwargs = dataloader_kwargs or {}

        for metadata_file, batch_size, loader_type, start in zip(metadata_files, batch_sizes, loader_types, start_idx):
            subdataset = CFDSubDataset(metadata_file, start, shuffle, split, flag)
            self.subdatasets.append(subdataset)
        
        for subdataset, batch_size, loader_type in zip(self.subdatasets, batch_sizes, loader_types):
            self.dataloaders.append(self.create_dataloader(subdataset, batch_size, loader_type, nodes_per_sample))

    def _sequence_collate(self, batch: list[list[Data]]):
        # batch is a list of sequences; each sequence is a list[Data] (len = seq_len).
        seq_len = len(batch[0])
        out = []
        for t in range(seq_len):
            graphs_at_t = [seq[t] for seq in batch]     # List[Data] at time t
            # Collater knows how to assemble per-time-step graphs into a single
            # batch while keeping multiscale connectivity consistent.
            batched = self.collater.collate(graphs_at_t)  # <-- Collater fixes multiscale indices and applies transforms
            out.append(batched)
        return out



    def create_dataloader(self, subdataset, batch_size, loader_type, nodes_per_sample):
        common_kwargs = {
            'batch_size': batch_size,
            'collate_fn': self._sequence_collate,   # <-- uses Collater(transform=...)
            # Sensible defaults keep backwards compatibility if the caller does
            # not override worker parameters.
            'num_workers': 1,
            'prefetch_factor': 1
            # DO NOT put 'shuffle' here for IterableDataset
        }

        if self._dataloader_kwargs:
            # Allow per-instance overrides such as num_workers/prefetch_factor.
            common_kwargs.update(self._dataloader_kwargs)

        if loader_type != 'default':
            raise ValueError(f"Unsupported loader type: {loader_type}")

        if isinstance(subdataset, torch.utils.data.IterableDataset):
            # No shuffle arg allowed; you already randomize order inside __iter__
            return TorchDataLoader(subdataset, **common_kwargs)
        else:
            # Map-style datasets can use shuffle
            return TorchDataLoader(subdataset, shuffle=getattr(subdataset, 'shuffle', False), **common_kwargs)



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

def create_cfd_datamodule(metadata_files, batch_sizes, loader_types, start_idx,
                          shuffle=False, split=1.0, flag='train',
                          nodes_per_sample=None,
                          collater_transform=None,
                          dataloader_kwargs=None):     # <— add this
    # The helper simply forwards the custom DataLoader kwargs to CFDDataset so
    # higher-level modules (Lightning data module) can decide how aggressively
    # to parallelise each stage.
    return CFDDataset(metadata_files, batch_sizes, loader_types, start_idx,
                      shuffle, split, flag, nodes_per_sample,
                      collater_transform=collater_transform,
                      dataloader_kwargs=dataloader_kwargs)


@torch.no_grad()
def estimate_mesh_scales(pos, edge_index, quantile=0.5):
    row, col = edge_index
    edge_len = (pos[col] - pos[row]).norm(dim=1)
    h = edge_len.quantile(quantile).item()  # median edge length
    rel_pos_scaling = [h, 2*h, 4*h, 8*h]
    return h, rel_pos_scaling

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