# D6 — Physical-space DGN backbone validation

## Purpose

D6 freezes the neural backbone contract for the physical-space HIT diffusion model. It does not change diffusion equations, the beta schedule, the hybrid loss, timestep sampling, training batching, or generation. Those are owned by later stages.

The objective is to prove that the existing DGN4AVBP implementation of `DiffusionGraphNet`, `MultiScaleGnn`, `InteractionNetwork`, `MeshDownMP`, and `MeshUpMP` is compatible with the frozen D1–D5 data/geometry contracts before any training is resumed.

## Reuse of the existing implementation

D6 does not introduce a replacement GNN. The existing implementation is preserved because it follows the original DGN4CFD structure:

- linear node and edge encoders;
- sinusoidal diffusion-step embedding;
- multiscale GNN encoder/bottleneck/decoder path;
- InteractionNetwork message passing;
- Guillard hierarchy transfers through `MeshDownMP` and `MeshUpMP`;
- learned reverse variance by decoding `2C` node channels and splitting them into two `C`-channel tensors.

Original reference implementation:

`tum-pbs/dgn4cfd/dgn4cfd/nn/diffusion/models/dgn.py`

## HIT input/output contract

The physical state contains five conservative channels:

```text
[rho, rhou, rhov, rhow, rhoE]
```

Therefore

```text
in_node_features = 5
```

The baseline is unconditional, so

```text
cond_node_features = 0
```

D3/D5 use vector relative positions in three dimensions, so

```text
cond_edge_features = 3
scalar_rel_pos = false
dim = 3
```

With learned variance enabled, the node decoder produces ten channels internally:

```text
5 epsilon channels + 5 variance-value channels
```

and `DiffusionGraphNet.forward` returns them as the tuple

```text
(epsilon, variance_value)
```

with each tensor shaped `[N, 5]`.

## Hierarchy scale selection

D5 measured the complete usable Guillard hierarchy rather than assuming a scale count:

```text
L1  35937 nodes
L2  17969
L3   5121
L4   1210
L5    241
L6     50
L7     12
L8      3
```

The canonical D6 baseline uses the prefix `L1..L7`.

The final three-node level remains persisted in D5 for future scale ablation, but it is excluded from the baseline because it collapses the global bottleneck to an extremely small graph. Level 7 already gives a graph-spanning coarse representation with 12 nodes and 70 directed edges.

This choice is also consistent with the original DGN4CFD scale strategy. The original Wing physical-space model, operating on a substantially smaller fine mesh, used six scales. D6 therefore uses seven scales for the larger 35,937-node HIT mesh rather than retaining the historical five-scale DGN4AVBP setting.

The selected message-passing depths are

```text
[2, 2, 2, 2, 2, 2, 2]
```

matching the two InteractionNetwork blocks per scale used by the original larger physical-space DGN example.

## Canonical architecture

`configs/model/dgn_hit_baseline.yaml` freezes:

```text
in_node_features   = 5
cond_node_features = 0
cond_edge_features = 3
depths             = [2,2,2,2,2,2,2]
fnns_depth         = 2
fnns_width         = 128
aggr               = sum
dropout            = 0.1
emb_width          = 512
dim                = 3
scalar_rel_pos     = false
learnable_variance = true
```

The D5 hierarchy is attached once per physical sample using the existing `attach_hierarchy_to_graph` adapter. D6 still validates one physical graph at a time. Correct multi-sample hierarchy index offsets remain owned by D9.

## Required validation

D6 unit tests require:

1. the canonical config to remain 5-channel input / 10-channel learned-variance output;
2. a full synthetic down -> bottleneck -> up forward pass;
3. final output shape `[N,5]` for both epsilon and variance-value tensors;
4. restoration of the fine `edge_index` and fine `batch` after the up path;
5. finite nonzero gradients through the complete backbone;
6. equivariance to a consistent node relabeling at every hierarchy level.

The real HIT validation additionally runs one full GPU forward pass using the persisted D5 hierarchy and the first frozen training sample after D3 preprocessing. It records parameter counts, output shapes, selected hierarchy levels, bottleneck size, and peak allocated CUDA memory.

## Explicit non-goals

D6 does not validate or change:

- DDPM forward/reverse equations;
- beta schedule;
- learned-range variance interpretation;
- HybridLoss or VLB weighting;
- timestep importance sampling;
- deterministic validation noise/timesteps;
- optimizer or learning-rate scheduler;
- computational batching;
- DDP;
- generation/sampling.

Those contracts begin in D7 and later stages.
