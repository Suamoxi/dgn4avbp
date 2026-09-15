# D3 — Physical nondimensionalization and train-only standardization

## Ownership

D1 remains the raw dimensional fixed-mesh snapshot dataset. D2 owns the persistent train/validation/test membership. D3 owns all numerical scaling used by the model.

The order is fixed:

1. dimensional AVBP state and geometry;
2. physical nondimensionalization from the HIT simulation reference scales;
3. statistical standardization of the five state channels only, fitted from the frozen D2 training split.

Validation and test samples never contribute to fitted statistics.

## Authoritative HIT reference scales

The reference file is `configs/data/hit_reference_scales.yaml`.

For `HIT_LES_FORCED`:

- `rho_ref = 1.17 kg/m^3`;
- `U_ref = 17.360947554785138 m/s`;
- `L_ref = 0.0005668079275737893 m`;
- `T_ref = 300 K`.

`U_ref` is the prescribed target one-component RMS velocity. `L_ref` is the periodic-box side length. The values are physical case inputs, not statistics inferred from the training targets.

## Physical nondimensionalization

The five conservative channels are ordered as

`[rho, rhou, rhov, rhow, rhoE]`.

They are transformed as

- `rho* = rho / rho_ref`;
- `rhou* = rhou / (rho_ref U_ref)`;
- `rhov* = rhov / (rho_ref U_ref)`;
- `rhow* = rhow / (rho_ref U_ref)`;
- `rhoE* = rhoE / (rho_ref U_ref^2)`.

Geometry is transformed as

- `x* = x / L_ref`;
- `e_ij* = (x_j - x_i) / L_ref`.

No statistical centering or variance normalization is applied to coordinates or edge geometry.

## Statistical standardization

For each nondimensional state channel `c`, D3 fits a population mean and standard deviation over the D2 training split.

The objective is explicitly sample-balanced. If sample `s` contains `N_s` nodes,

`mu_c = (1/S) sum_s [(1/N_s) sum_i x*_{s,i,c}]`.

The population second moment is accumulated with the same sample weighting, and

`sigma_c = sqrt(E[(x*_c)^2] - mu_c^2)`.

The model state is then

`z_c = (x*_c - mu_c) / sigma_c`.

For the current fixed HIT mesh every sample has the same node count, so sample-balanced and global node-balanced moments are numerically identical. The implementation nevertheless encodes equal physical-sample weighting explicitly because that is the intended training objective.

## Persistent artifact

`scripts/fit_d3_preprocessing.py` writes `artifacts/d3_preprocessing_manifest.json` containing:

- the full D2 dataset fingerprint;
- `fit_split = train`;
- the exact number of fitting samples;
- a separate SHA-256 fingerprint of the D2 training file list;
- the complete physical reference configuration;
- the five fitted nondimensional means and population standard deviations;
- the formulas and scaling policy.

A preprocessing artifact is invalid if its dataset fingerprint or training-file fingerprint does not match the D2 manifest.

## Validation requirements

D3 is accepted only if:

- physical state scaling round-trips to the dimensional state;
- geometry scaling round-trips to the dimensional mesh;
- the nondimensional HIT box span is `[1, 1, 1]` within numerical tolerance;
- edge attributes equal differences of nondimensional node positions;
- recomputing moments over all D2 training samples after standardization gives channel mean `0` and population standard deviation `1` within numerical tolerance;
- modifying validation/test values cannot change fitted training statistics in unit tests.

## Non-goals

D3 does not alter:

- train/validation/test membership;
- mesh connectivity;
- periodic closure;
- multiscale coarsening;
- model architecture;
- diffusion schedule;
- batching or DDP.
