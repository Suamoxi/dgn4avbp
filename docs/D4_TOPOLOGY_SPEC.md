# D4 — Native HIT topology validation

## Purpose

D4 validates the fixed HIT mesh topology used by the physical-space DGN baseline. It does not modify the mesh and does not add periodic closure.

The active graph must represent only the native hexahedral nearest-neighbour edges present in `Connectivity/hex->node`.

## Local hexahedron edge mappings

The current D1 loader uses the standard local cube edge list

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

Although the order looks different, both lists contain exactly the same 12 undirected local cube edges. D4 verifies both the local-set equivalence and that they produce the same coalesced global directed graph on the real HIT mesh.

## HIT mesh contract

For the validated HIT case, D4 requires the raw coordinates to form a complete Cartesian tensor-product grid. From D1/D3 this is expected to be

```text
33 x 33 x 33 nodes
32 x 32 x 32 hexahedral cells
```

with physical span `L_ref` in each direction.

With no periodic closure, the native grid spacing is

```text
h = L_ref / 32
```

and the nondimensional spacing after D3 geometry scaling is

```text
h* = 1 / 32 = 0.03125.
```

## Edge geometry

Every active graph edge must be an axis-aligned nearest-neighbour mesh edge:

- exactly one coordinate changes;
- its magnitude equals the corresponding native grid spacing;
- the other two coordinate differences are zero within numerical tolerance;
- no edge spans more than one grid interval;
- no edge directly connects opposite box faces.

`edge_attr` must equal `pos[j] - pos[i]` exactly in raw dimensional coordinates.

The D3 preprocessed graph must satisfy

```text
pos* = pos / L_ref
edge_attr* = edge_attr / L_ref
```

with no statistical standardization of coordinates or edge geometry.

## Expected nonperiodic degree distribution

For a `33^3` Cartesian grid with only axial nearest-neighbour edges:

- 8 corner nodes have degree 3;
- 372 edge-interior nodes have degree 4;
- 5766 face-interior nodes have degree 5;
- 29791 volume-interior nodes have degree 6.

The degree sum is therefore 209088, which is also the number of stored directed edges because the PyG graph contains both directions.

A periodic closure would instead alter boundary degrees and introduce direct opposite-face edges. D4 explicitly requires both effects to be absent.

## Persistence

`scripts/validate_d4_topology.py` writes

```text
artifacts/d4_topology_manifest.json
```

containing the inferred Cartesian grid, raw and nondimensional spacing, degree histogram, edge-geometry diagnostics, local-mapping equivalence, D2 dataset fingerprint, and D3 training-file fingerprint.

## Non-goals

D4 does not:

- add periodic edges;
- apply minimum-image relative positions;
- construct the multiscale Guillard hierarchy;
- change the DGN architecture;
- change diffusion math;
- change batching or training.

Periodic topology remains a future controlled ablation. D5 owns multiscale hierarchy construction and validation.
