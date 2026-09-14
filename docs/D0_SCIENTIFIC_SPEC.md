# D0 — Frozen scientific specification

This document freezes the scientific contract for the first trustworthy DGN4AVBP baseline.

## Scientific objective

The baseline learns the equilibrium distribution of HIT CFD states on one fixed AVBP mesh using a physical-space Diffusion Graph Network (DGN).

The baseline is **not** a temporal forecasting model. File ordering or filename timestamps may be retained as provenance, but temporal windows, previous states, and future states are not model inputs.

## Physical sample

One AVBP HDF5 solution file is exactly one physical sample.

For a dataset containing `K` solution files:

- dataset length is exactly `K`;
- every solution file appears exactly once before any train/validation/test split is applied;
- no sequence/window construction is allowed;
- no snapshot may be discarded because of a sequence length;
- no snapshot may be duplicated implicitly by an iterable loader.

## State definition

One CFD mesh node is one graph vertex.

The state has exactly five conservative channels, in this order:

1. `rho`
2. `rhou`
3. `rhov`
4. `rhow`
5. `rhoE`

The source HDF5 paths are:

- `GaseousPhase/rho`
- `GaseousPhase/rhou`
- `GaseousPhase/rhov`
- `GaseousPhase/rhow`
- `GaseousPhase/rhoE`

Derived velocity quantities such as `u = rhou / rho` are evaluation quantities unless a later controlled experiment explicitly changes the task.

## Mesh definition

The baseline uses one shared fixed AVBP mesh for all samples.

Coordinates are read from:

- `Coordinates/x`
- `Coordinates/y`
- `Coordinates/z`

Hexahedral connectivity is read from:

- `Connectivity/hex->node`

The mesh is loaded once and reused by all snapshots.

### Periodicity policy

Periodicity is **not enforced in the baseline**.

Only edges implied by the native hexahedral connectivity are allowed. Nodes on opposite faces are not connected merely because the physical HIT case is periodic. Periodic closure is reserved for a later controlled experiment.

D4 will therefore validate the native topology rather than add periodic edges.

## Preprocessing ownership

The intended preprocessing order is:

`dimensional CFD state -> physical nondimensionalization -> train-only statistical standardization`

D1 exposes the raw dimensional state only.

Nondimensionalization and statistical standardization are introduced in D3. Statistical quantities must be fitted on the training split only.

## Model scope

The first baseline is the physical-space DGN.

The VGAE and latent DGN (LDGN) are out of scope until the physical-space DGN has passed the HIT distribution benchmark. They will be treated as separate experiments so VGAE reconstruction error is measurable independently from diffusion error.

## Training scope

The initial baseline targets one GPU only.

DDP is explicitly deferred. No distributed-training-specific scaling or synchronization is required for the first baseline.

## Iteration boundaries

- D0: this frozen scientific specification.
- D1: one HDF5 snapshot = one dataset sample, fixed mesh loaded once, file/data manifest, validation gate.
- D2: persistent train/validation/test split policy.
- D3: physical nondimensionalization and train-only statistical scaling.
- D4: native HIT topology validation; no periodic closure.
- D5: reusable multiscale hierarchy.
- D6+: DGN/diffusion/training correctness and physical benchmarking.

## D0 acceptance criteria

D0 is complete when the code and documentation agree on all of the following:

- one HDF5 file is one physical sample;
- the state is `[rho, rhou, rhov, rhow, rhoE]`;
- one shared fixed mesh is used;
- no temporal conditioning is used;
- no periodic edges are introduced;
- the first baseline is physical-space DGN;
- the first baseline is single-GPU.
