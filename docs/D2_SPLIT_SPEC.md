# D2 — Persistent train/validation/test split

## Dataset basis

D2 operates on the D1 fixed-mesh HIT snapshot dataset:

- one numbered `solut_hit_*.h5` file = one physical sample;
- `last_solution.h5` is excluded because it is byte-identical to `solut_hit_00013342.h5`;
- the remaining dataset contains 1261 unique numbered snapshots;
- snapshots belong to one time-ordered HIT trajectory.

## Split strategy

The split is deterministic and chronological:

1. parse the integer suffix from each numbered solution file;
2. sort snapshots by that integer;
3. assign one contiguous block to training, then validation, then test;
4. never randomly interleave snapshots from this trajectory across splits.

Default requested ratios are 0.8 / 0.1 / 0.1. Integer sample counts use the largest-remainder method so every snapshot is assigned exactly once.

For the current 1261-snapshot dataset the expected counts are:

- train: 1009;
- validation: 126;
- test: 126.

With the current file set, the expected iteration ranges are:

- train: 760 through 10830;
- validation: 10840 through 12090;
- test: 12100 through 13342.

The nonstandard numbered snapshots 4464 and 13342 remain valid physical samples and are retained in chronological order.

## Persistence and leakage control

`scripts/create_d2_split.py` writes `artifacts/d2_split_manifest.json`. The manifest stores:

- the exact file path assigned to each split;
- requested and realized split ratios;
- sample counts;
- snapshot iteration ranges;
- boundary iteration gaps;
- a SHA-256 fingerprint of the complete ordered dataset file list.

Any later stage must validate the manifest against the current dataset file list before using it. D3 normalization statistics must be fitted from the frozen training split only.

## Non-goals

D2 does not:

- nondimensionalize fields;
- standardize fields;
- modify mesh topology;
- add periodic edges;
- configure batching;
- configure the diffusion process or model.
