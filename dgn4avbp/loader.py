from typing import List

import torch.utils.data
from torch_geometric.data import Batch, Data
from torchvision import transforms


class Collater(object):
    """Collate physical graphs with the multiscale indices expected by DGN.

    PyG already increments every attribute whose name contains ``index`` by the
    cumulative number of fine-level nodes. Therefore, for ``edge_index_L`` we
    only pre-compensate by

        cumulative_nodes_L - cumulative_nodes_1

    so PyG's native increment leaves the final indices in the level-L node
    space. Parent maps ``idx{L-1}_to_idx{L}`` do not contain ``index`` and thus
    require the complete cumulative level-L offset explicitly. ``batch_L`` is
    handled natively by PyG because its key contains ``batch``. Hexahedral
    ``cells`` also need an explicit fine-level node offset because PyG does not
    recognize that attribute as connectivity.
    """

    def __init__(self, transform: transforms.Compose = None):
        self.transform = transform

    def collate(self, batch: List[Data]):
        if len(batch) == 0:
            return None

        elem = batch[0]
        max_level = 1
        while hasattr(elem, f"edge_index_{max_level + 1}"):
            max_level += 1

        for graph in batch:
            for level in range(2, max_level + 1):
                required = (
                    f"pos_{level}",
                    f"edge_index_{level}",
                    f"idx{level - 1}_to_idx{level}",
                    f"batch_{level}",
                )
                missing = [name for name in required if not hasattr(graph, name)]
                if missing:
                    raise ValueError(
                        f"All graphs in a hierarchy batch must expose the same levels; "
                        f"missing {missing}."
                    )
            if hasattr(graph, f"edge_index_{max_level + 1}"):
                raise ValueError("All graphs in a hierarchy batch must expose the same number of levels.")

        # ``cells`` stores fine-node connectivity but its key does not activate
        # PyG's index increment heuristic.
        has_cells = [hasattr(graph, "cells") for graph in batch]
        if any(has_cells) and not all(has_cells):
            raise ValueError("Either every graph in a batch must expose cells or none of them may do so.")
        if all(has_cells):
            cumulative_fine_nodes = int(elem.num_nodes)
            for graph in batch[1:]:
                graph.cells = graph.cells + cumulative_fine_nodes
                cumulative_fine_nodes += int(graph.num_nodes)

        # Correct hierarchy indices before delegating the actual concatenation
        # to PyG. Graph 0 needs no correction; cumulative counts describe all
        # previously seen graphs.
        for level in range(2, max_level + 1):
            cumulative_fine_nodes = int(elem.num_nodes)
            cumulative_level_nodes = int(getattr(elem, f"pos_{level}").size(0))

            for graph in batch[1:]:
                edge_key = f"edge_index_{level}"
                parent_key = f"idx{level - 1}_to_idx{level}"

                # PyG will subsequently add cumulative_fine_nodes because
                # ``edge_index`` contains the substring "index".
                edge_compensation = cumulative_level_nodes - cumulative_fine_nodes
                setattr(graph, edge_key, getattr(graph, edge_key) + edge_compensation)

                # ``idx...`` does not trigger PyG's index increment heuristic.
                setattr(
                    graph,
                    parent_key,
                    getattr(graph, parent_key) + cumulative_level_nodes,
                )

                cumulative_fine_nodes += int(graph.num_nodes)
                cumulative_level_nodes += int(getattr(graph, f"pos_{level}").size(0))

        out = Batch.from_data_list(batch)
        return self.transform(out) if self.transform is not None else out

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    """PyG-like DataLoader using the hierarchy-aware ``Collater`` above."""

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        transform: transforms.Compose = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(transform),
            **kwargs,
        )
