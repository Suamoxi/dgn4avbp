# D9 — hierarchy-aware multi-sample batching

D9 freezes the computational batching contract for the physical-space HIT DGN.
It does **not** choose a production training batch size and does not introduce
DDP. Those decisions remain downstream.

## Why a custom collation correction is required

PyTorch Geometric batches ordinary `edge_index` correctly because its default
`Data.__inc__` increments every key containing `index` by the cumulative number
of fine-level nodes.

The DGN hierarchy contains several different node index spaces:

- `edge_index` indexes level-1 nodes;
- `edge_index_L` indexes level-L nodes;
- `idx{L-1}_to_idx{L}` maps level-(L-1) nodes to level-L nodes;
- `batch_L` identifies the physical graph for each level-L node.

For graph `g` in a computational batch, define

- `N_1^<g` = number of fine nodes in all previous graphs;
- `N_L^<g` = number of level-L nodes in all previous graphs.

PyG will automatically add `N_1^<g` to `edge_index_L` because the key contains
`index`. Therefore D9 pre-compensates only by

```text
N_L^<g - N_1^<g
```

so the final increment becomes exactly `N_L^<g`.

`idx{L-1}_to_idx{L}` does not contain the substring `index`, so D9 explicitly
adds the full destination-level offset `N_L^<g`.

`batch_L` is **not** manually shifted: PyG recognizes keys containing `batch`
and increments them natively by graph id.

The AVBP `cells` tensor also stores level-1 node indices, but its key does not
activate PyG's index heuristic. D9 therefore explicitly offsets each later cell
block by `N_1^<g`.

## Invariants

For every hierarchy level `L` in a valid batch:

1. node tensors are concatenated along the node dimension;
2. all `edge_index_L` values lie in that level's batched node space;
3. an edge never connects nodes belonging to different physical graphs;
4. `batch_L` contains the correct physical graph id for every level-L node;
5. every parent relation preserves graph identity:

   ```text
   batch_{L-1}[i] == batch_L[idx{L-1}_to_idx{L}[i]]
   ```

6. fine-level AVBP cell connectivity remains inside the corresponding graph's
   fine-node range.

## Model equivalence

In `eval()` mode, batching must be purely computational. Given the same model,
same hierarchy, same diffusion timesteps, and same noisy fields,

```text
DGN(batch(graph_a, graph_b))
```

must equal, up to floating-point tolerance,

```text
concat(DGN(graph_a), DGN(graph_b)).
```

This is validated for both the epsilon and learned-range variance heads.

## Deterministic validation

D8 assigned every validation snapshot a fixed timestep and Gaussian realization
from `(base_seed, sample_id)`. D9 extends this contract to computational batches.
A snapshot must receive exactly the same timestep and noise tensor whether it is
validated alone or inside a batch with other snapshots.

## Scope

D9 validates hierarchy-aware batching for `B > 1` on one process / one GPU.
It does not define:

- the production training batch size;
- distributed synchronization or DDP (D10);
- sampler/RNG checkpoint state (D11);
- the production training or generation loops (D12).
