# D11 — checkpointing and reproducibility

D11 freezes the checkpoint contract used by the single-process physical-space HIT DGN before the production training loop is assembled in D12.

## Exact-resume scope

The hard D11 guarantee is **exact resume at an epoch boundary on one process / one GPU**.

A D11 checkpoint stores:

- model state;
- optimizer state;
- optional learning-rate scheduler state;
- optional AMP gradient-scaler state;
- diffusion timestep sampler state;
- epoch;
- global optimizer step;
- batch-in-epoch counter for provenance;
- Python RNG state;
- NumPy RNG state;
- PyTorch CPU RNG state;
- all visible PyTorch CUDA RNG states;
- an optional dedicated CPU `torch.Generator` state for shuffled data order;
- arbitrary run metadata/fingerprints.

The loss-second-moment sampler therefore resumes with the same `loss_history` and `loss_counts`; it does not warm up again after restart.

## Why epoch-boundary resume is the hard contract

For a standard shuffled PyTorch `DataLoader`, the full sample permutation is normally materialized when the iterator is created. Capturing only a batch counter and generator state in the middle of that iterator is insufficient to reconstruct the already-created permutation exactly.

D11 therefore does **not** claim exact mid-epoch shuffled-data resume. Production D12 checkpoints should be written at epoch boundaries. Exact mid-epoch continuation would require a stateful data sampler or serialization of the active epoch permutation and cursor.

## Dedicated data generator

D12 should construct the shuffled training loader with an explicit CPU `torch.Generator` and pass that same generator to D11 save/load functions. This separates data-order reproducibility from unrelated global PyTorch random draws.

## Atomic save

Checkpoints are first written to `<checkpoint>.tmp` and then moved into place with `os.replace`. A partial write therefore does not overwrite the last valid checkpoint path.

## Compatibility checks

Resume fails loudly when:

- checkpoint format versions differ;
- timestep sampler classes differ;
- diffusion step counts differ;
- loss-second-moment sampler history length differs;
- loss-second-moment uniform probability differs;
- scheduler presence differs between checkpoint and resumed run;
- AMP scaler presence differs between checkpoint and resumed run;
- a stored data-generator state is present but no generator is supplied;
- CUDA RNG state exists but the visible CUDA device count is incompatible.

## Validation

D11 unit tests require an uninterrupted CPU training continuation and a save/restore continuation to produce exactly the same:

- Python, NumPy, global Torch and data-generator random draws;
- sampled diffusion timesteps and importance weights;
- loss;
- model parameters after the next update;
- Adam state;
- scheduler state;
- loss-second-moment sampler state.

The Slurm validator additionally checks CUDA RNG restoration and AMP scaler restoration on the cluster GPU.

## Deferred

D11 does not implement:

- DDP checkpoint synchronization/resume (D10, intentionally deferred);
- the production data/training loop (D12);
- generation checkpoint selection (D12+);
- exact mid-epoch shuffled-data resume.
