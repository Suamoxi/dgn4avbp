# D12 — physical-space HIT training, validation, and generation pipeline

D12 assembles the frozen D1–D9 and D11 contracts into the first production
single-GPU pipeline for the physical-space HIT Diffusion Graph Network.

D10 (DDP) remains intentionally deferred.

## Training

The production trainer uses:

- the frozen D2 contiguous train split;
- D3 nondimensionalization and train-only standardization;
- the persisted D5 Guillard hierarchy;
- the D6 seven-level DGN backbone;
- the D7 1000-step linear Improved-DDPM process and canonical hybrid loss;
- the D8 loss-second-moment timestep sampler and `1/(T p(t))` importance weight;
- the corrected D9 hierarchy-aware multi-sample collater;
- D11 epoch-boundary exact-resume checkpoints.

The initial baseline retains the main optimizer choices used by the released
DGN4CFD physical-space examples where compatible with the HIT contract:

- Adam;
- learning rate `1e-4`;
- gradient clipping at L2 norm `1`;
- ReduceLROnPlateau with factor `0.1` and patience `250`;
- scheduler monitor: weighted training objective.

The validated computational batch size is initially `2`. D12 does not claim
that this is the throughput-optimal batch size.

The initial baseline is FP32. Mixed precision is a later runtime ablation rather
than a silent change to the frozen baseline.

## Validation

Validation is not randomly noised at every epoch. For each frozen validation
sample, D8 deterministically maps `(base_seed, sample_id)` to one timestep and
one Gaussian realization. Therefore validation is stable across epochs and
iteration order.

The monitored quantity is the same canonical per-graph hybrid loss used by
training before timestep importance weighting. It is a stable monitoring metric,
not an exact sum over all 1000 diffusion timesteps.

Validation never updates the training loss-second-moment sampler.

## Checkpoints

Production checkpoints are written at epoch boundaries and contain:

- model state;
- optimizer state;
- scheduler state;
- optional AMP scaler state;
- loss-second-moment sampler history;
- epoch and global step;
- Python/NumPy/PyTorch CPU/CUDA RNG state;
- dedicated training DataLoader generator state.

The exact-resume claim is the D11 contract: one process, one GPU, epoch-boundary
resume. Arbitrary mid-epoch shuffled-data resume is not claimed.

## Generation

The generation task is unconditional equilibrium HIT generation. A dataset
sample is used only as the carrier of the fixed mesh and hierarchy; its target
field is not used to condition generation.

The production default is the complete 1000-step ancestral DDPM chain. D12's
Slurm integration test uses a short linearly spaced subset only to validate the
end-to-end code path without turning a smoke test into a full generation job.

Generated samples are saved in three state representations:

1. standardized network space;
2. nondimensional conservative variables after inverse D3 z-scoring;
3. dimensional conservative variables `[rho, rhou, rhov, rhow, rhoE]` after
   restoring the D3 HIT reference scales.

The output also records nondimensional/dimensional positions and cell
connectivity.

## D12 integration gate

The real-mesh D12 Slurm check performs:

1. all D1–D9/D11 unit tests plus D12 unit tests;
2. one real `B=2` HIT training update with backward/Adam/gradient clipping;
3. one real `B=2` deterministic validation batch;
4. a D11 checkpoint save and exact model-state restore;
5. one real HIT unconditional generation with a short spaced ancestral chain;
6. inverse preprocessing to nondimensional and dimensional AVBP states;
7. finite/shape checks and CUDA-memory reporting.

The smoke-generation artifact is explicitly not a scientific sample: the model
has received only one optimizer update.
