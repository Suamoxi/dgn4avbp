# D8 — Timestep sampling and deterministic validation

D8 freezes how diffusion timesteps are selected during training and how validation corruption is generated. It does not change the D7 diffusion equations or the D6 neural backbone.

## Training sampler

The canonical physical-space HIT baseline uses the Improved-DDPM loss-second-moment sampler already present in `dgn4avbp.step_sampler.ImportanceStepSampler`:

- 1000 diffusion steps,
- history length 10 per timestep,
- uniform mixture probability 0.001,
- warm-up remains uniform until every timestep has 10 recorded losses,
- post-warm-up sampling weight is proportional to the square root of the empirical second moment of the per-sample loss,
- the objective correction is `1 / (T p(t))`.

The sampler is updated with the unweighted per-sample hybrid loss. The importance weight is applied only afterwards when constructing the minibatch objective.

D8 makes one minimal implementation correction: sampler history is stored in NumPy, therefore CUDA timestep/loss tensors are explicitly detached and moved to CPU before they index/update NumPy history arrays.

## Validation corruption

Validation must not resample random timesteps and Gaussian noise every epoch. Each frozen validation sample is keyed by its existing `sample_id`.

For a base seed `s` and sample identifier `id`, SHA-256 is used to derive two independent stable integers:

1. a timestep key, reduced modulo `T`,
2. a Gaussian-noise seed.

The Gaussian realization is generated with a CPU `torch.Generator` and then moved to the field device. Consequently the validation corruption depends only on `(base_seed, sample_id)`, not on epoch number, validation iteration order, Python hash randomization, or CPU/GPU placement.

D8 intentionally validates one physical graph at a time. Hierarchy-aware batching is deferred to D9.

## Interpretation

With one fixed hashed timestep per validation snapshot, the validation loss is a stable monitoring metric. It is not claimed to be the exact full sum over all 1000 diffusion timesteps. The policy is chosen to eliminate validation noise while preserving coverage across the frozen validation set.

Validation does not update or otherwise mutate the training loss-second-moment sampler.

## Non-goals

D8 does not implement:

- hierarchy-aware multi-sample batching (D9),
- DDP sampler synchronization (D10),
- checkpointing sampler/RNG states (D11),
- the final training driver or generation workflow.
