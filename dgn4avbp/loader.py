from typing import List
import torch.utils.data
from torch_geometric.data import Data, Batch
from torchvision import transforms

class Collater(object):
    """
    Collate that correctly offsets multi-scale indices:
      - edge_index_k        (k >= 2) : + cumulative #nodes at level k
      - idx{k-1}_to_idx{k}  (k >= 2) : + cumulative #nodes at level k
      - batch_k             (k >= 2) : + graph_id (0..B-1) so it behaves like PyG's .batch
    """
    def __init__(self, transform: transforms.Compose = None):
        self.transform = transform

    def collate(self, batch: List[Data]):
        if len(batch) == 0:
            return None

        # --- prepare per-level node offsets and graph-id counter ---
        # For each level k>=2, offsets[k] = total #nodes at level k in all *previous* graphs
        level_node_offsets = {}   # dict[int -> int]
        graph_id = 0              # this will be added to batch_k

        # probe how many levels exist from the first elem
        elem = batch[0]
        max_level = 1
        while hasattr(elem, f'edge_index_{max_level+1}'):
            max_level += 1

        # --- walk through the graphs in the incoming mini-batch ---
        for i, g in enumerate(batch):
            # initialize offsets for all levels seen so far
            for L in range(2, max_level+1):
                if L not in level_node_offsets:
                    level_node_offsets[L] = 0

            # shift all multi-scale tensors for this graph
            for L in range(2, max_level+1):
                # edge_index_L lives in the index space of level-L nodes -> add node offset at L
                ek = f'edge_index_{L}'
                if hasattr(g, ek):
                    setattr(g, ek, getattr(g, ek) + level_node_offsets[L])

                # idx{L-1}_to_idx{L} maps level-(L-1) nodes to level-L nodes -> shift values by offset at L
                mk = f'idx{L-1}_to_idx{L}'
                if hasattr(g, mk):
                    setattr(g, mk, getattr(g, mk) + level_node_offsets[L])

                # batch_L is per-node graph id at level L -> bump by current graph_id
                bk = f'batch_{L}'
                if hasattr(g, bk):
                    setattr(g, bk, getattr(g, bk) + graph_id)

            # after shifting, update the cumulative node counts for each level
            for L in range(2, max_level+1):
                pk = f'pos_{L}'
                if hasattr(g, pk):
                    level_node_offsets[L] += int(getattr(g, pk).size(0))

            graph_id += 1

        # --- hand over to PyG; it will handle level-1 (.edge_index and .batch) automatically ---
        out = Batch.from_data_list(batch)
        return self.transform(out) if self.transform is not None else out

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    """ PyG-like DataLoader using the custom Collater above. """
    def __init__(self, dataset, batch_size=1, shuffle=False, transform: transforms.Compose=None, **kwargs):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        super().__init__(dataset, batch_size, shuffle, collate_fn=Collater(transform), **kwargs)
