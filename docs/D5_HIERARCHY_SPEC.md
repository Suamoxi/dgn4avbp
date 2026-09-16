# D5 — Fixed-mesh Guillard hierarchy

## Purpose

D5 builds the multiscale graph hierarchy once for the fixed mesh, persists it, reloads it, and reuses the same hierarchy for every physical snapshot.

The hierarchy is a property of the mesh geometry/topology. It must not be recomputed independently for every CFD state.

D5 does not choose the final DGN architecture. It measures the complete usable hierarchy first; D6 will select a prefix that matches the model depth.

## Inputs frozen by earlier stages

D5 consumes:

- D1: fixed native AVBP mesh and one-HDF5-per-snapshot dataset;
- D2: persistent dataset/split fingerprint;
- D3: nondimensional geometry `x* = x / L_ref`;
- D4: `native_hexahedral_connectivity` graph contract.

No periodic closure is added.

## Reuse of the existing Guillard implementation

D5 does not introduce a second coarsening algorithm. It reuses the existing

```text
dgn4avbp.transform_locals.guillard_coarsening
dgn4avbp.transform_locals.pool_edges
```

logic.

At every transition `l -> l+1`:

1. Guillard selects the surviving coarse nodes;
2. every high-resolution node receives a low-resolution parent index;
3. existing edges are pooled through that parent map and coalesced;
4. no new periodic or geometric-neighbour search is performed.

## Geometry convention

All hierarchy geometry uses one convention:

```text
x_l = x_l,dimensional / L_ref
edge_attr_l = x_l[j] - x_l[i]
e_l,l+1 = x_l+1[parent(i)] - x_l[i]
```

There is no `1/h` scaling on level 1 and no independent `2h`, `4h`, ... scaling on deeper levels.

The physical coarsening naturally makes coarse edges longer in the same nondimensional coordinate system.

This replaces the historical preprocessing path in which the fine edge vectors were scaled by `1/h` before `MeshCoarsening`, while coarse edge vectors were recomputed from unscaled coordinates when `rel_pos_scaling=None`.

## Number of levels

D5 does not assume `num_scales=5`.

With

```yaml
max_levels: null
```

successive Guillard levels are built until another level would contain fewer than two nodes, contain no edges, or fail to reduce the node count.

The manifest records for every level:

```text
N_l = number of nodes
E_l = number of directed edges
min / mean / max edge length in x/L_ref units
```

D6 will inspect this measured sequence and choose the model hierarchy prefix.

An explicit `max_levels` remains available for tests or controlled ablations.

## Persisted artifacts

D5 writes:

```text
artifacts/d5_hierarchy.pt
artifacts/d5_hierarchy_manifest.json
```

The `.pt` file stores the hierarchy tensors:

- positions for every level;
- `edge_index` and `edge_attr` for every level;
- Guillard coarse masks;
- high-resolution to low-resolution parent maps;
- fine-to-coarse displacement vectors.

The JSON manifest stores provenance, level sizes, edge-length statistics and a logical SHA-256 fingerprint over all hierarchy tensors.

The builder immediately reloads both files and validates them. Persistence is therefore part of the D5 gate, not an unchecked output step.

## Compatibility with `MultiScaleGnn`

`attach_hierarchy_to_graph` exposes the persisted tensors with the attribute names currently expected by `MultiScaleGnn` and `MeshDownMP`, e.g.

```text
coarse_mask_2
pos_2
idx1_to_idx2
edge_index_2
edge_attr_2
e_12
batch_2
```

and similarly for deeper levels.

The hierarchy is attached to a single physical graph in D5. Correct replication/index offsets for computational batches are intentionally deferred to D9; D5 must not silently rely on PyG's default batching rules for the custom multiscale indices.

## Validation contract

For every level D5 requires:

- node count strictly decreases;
- non-empty connectivity;
- valid node indices;
- no self-loops;
- exact bidirectionality;
- `edge_attr = pos[j] - pos[i]`;
- finite, non-zero geometric edge vectors.

For every transition D5 requires:

- the coarse mask selects exactly the next-level nodes;
- next-level coordinates are exactly the selected previous-level coordinates;
- all parent indices are in range;
- `e_l,l+1` equals the direct parent displacement in the same coordinate convention;
- pooled connectivity is exactly reproducible from the previous level and parent map.

Building the hierarchy twice from the same mesh must produce the same logical fingerprint.

## Non-goals

D5 does not:

- select the final number of DGN scales;
- change the message-passing backbone;
- change the diffusion task;
- add periodicity;
- add edge pruning to the baseline;
- define multi-sample batching.

D6 owns backbone/hierarchy-prefix compatibility. D9 owns computational batching.
