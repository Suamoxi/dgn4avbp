# D4 — Native mesh topology validation

## Purpose

D4 validates the topology used by the physical-space DGN baseline without assuming that an AVBP mesh is Cartesian, uniform, or structured.

The hard D4 contract is:

```text
native AVBP hexahedral connectivity
        -> graph topology
        -> relative edge geometry
```

For the current D1 data path, one mesh node is one graph vertex and `Connectivity/hex->node` is the authoritative topology source. D4 does not modify that mesh and does not add periodic closure.

The current HIT case happens to be a structured box. Those Cartesian properties are useful diagnostics for this dataset, but they are not requirements of DGN4AVBP and are not part of the generic D4 pass/fail criterion.

## Core topology contract

`validate_native_hex_topology` verifies the following for any hexahedral mesh handled by the current D1 loader:

- node coordinates are finite;
- hexahedral connectivity has shape `[num_cells, 8]`;
- connectivity indices lie inside the mesh node range;
- a cell does not repeat a local node index;
- the active `edge_index` is exactly the coalesced bidirectional graph derived from native connectivity;
- the graph contains no self-loops;
- the graph is exactly bidirectional;
- `edge_attr` has the correct shape and is finite;
- `edge_attr = pos[j] - pos[i]` exactly in raw dimensional coordinates;
- graph edges do not have zero geometric length;
- no edge beyond those generated from `Connectivity/hex->node` is added by D4/D1.

No requirement is placed on:

- Cartesian alignment;
- uniform spacing;
- box-shaped domains;
- a specific node degree distribution;
- a specific number of nodes or cells;
- tensor-product coordinate planes.

This is the topology contract that future arbitrary AVBP hexahedral meshes must satisfy.

## Local hexahedron edge mappings

The current D1 loader uses

```text
(0,1) (1,2) (2,3) (3,0)
(4,5) (5,6) (6,7) (7,4)
(0,4) (1,5) (2,6) (3,7)
```

The legacy `utils.convert_element_to_coo` code used

```text
(0,1) (1,5) (5,4) (4,0)
(3,2) (2,6) (6,7) (7,3)
(0,3) (1,2) (5,6) (4,7)
```

Although the order looks different, both lists contain the same 12 undirected local cube edges. D4 verifies both the local-set equivalence and that both lists generate the same coalesced global directed graph.

## D3 geometry interface

The D3 preprocessed graph must still satisfy the hard interface contract

```text
pos* = pos / L_ref
edge_attr* = edge_attr / L_ref
```

with no statistical standardization of coordinates or edge geometry.

The current reference configuration must also match the reference configuration frozen in the D3 preprocessing manifest.

## Optional HIT Cartesian diagnostics

Because `HIT_LES_FORCED` is expected to be a regular box, D4 also runs a separate diagnostic via `run_hit_cartesian_diagnostics`.

For this case only, the diagnostic examines:

- whether coordinates form a complete Cartesian tensor-product grid;
- coordinate-plane counts;
- physical span compared with `L_ref`;
- uniform grid spacing;
- expected Cartesian edge count;
- expected nonperiodic degree histogram;
- whether every edge is one axis-aligned nearest-neighbour interval;
- whether direct cross-box or opposite-face edges appear.

For the known HIT dataset, the expected values are approximately

```text
33 x 33 x 33 nodes
32 x 32 x 32 hexahedral cells
h = L_ref / 32
h* = 1 / 32 = 0.03125
```

and the expected nonperiodic degree distribution is

```text
degree 3:     8
degree 4:   372
degree 5:  5766
degree 6: 29791
```

These checks are **diagnostic only**. A failure is written into the D4 manifest but does not invalidate the generic topology contract. This separation is intentional: an unstructured or deformed AVBP mesh must be allowed to pass D4 when its native connectivity and graph construction are correct.

Coordinate-plane clustering used by this optional diagnostic is tolerant to tiny floating-point roundoff so that numerically equivalent planes are not counted twice.

## Persistence

`scripts/validate_d4_topology.py` writes

```text
artifacts/d4_topology_manifest.json
```

The manifest contains:

- generic native-connectivity validation results;
- observed degree histogram;
- generic edge-length diagnostics;
- local-mapping equivalence;
- D2 dataset fingerprint;
- D3 training-file fingerprint;
- D3 geometry consistency;
- optional HIT Cartesian diagnostics and their `passed` or `failed` status.

## Non-goals

D4 does not:

- add periodic edges;
- apply minimum-image relative positions;
- require a Cartesian grid;
- require uniform spacing;
- construct the multiscale Guillard hierarchy;
- change the DGN architecture;
- change diffusion math;
- change batching or training.

Periodic topology remains a future controlled ablation. D5 owns multiscale hierarchy construction and validation.
