# D13 — 3-D HIT generation distribution benchmark

D13 evaluates the scientific distribution represented by generated HIT snapshots. It is post-processing only: it consumes existing `sample_*.pt` generation artifacts and never reruns training or diffusion sampling.

## Comparison semantics

The canonical D0 task is unconditional equilibrium generation. A generated snapshot therefore has no unique test target. D13 compares

```text
generated population  <->  frozen D2 test population
```

as **unpaired populations**. `seed` and `sample_index` identify random generations only; they never create a generated-to-reference target pair. D13 consequently does not report pointwise generated-vs-test MSE, MAE, or difference fields.

## State space

The benchmark operates in the D3 physically nondimensionalized conservative-variable space

```text
[rho, rhou, rhov, rhow, rhoE]
```

before statistical z-scoring. It reports conservative-channel marginal statistics as well as derived velocity and energy quantities.

For every conservative channel D13 reports pooled mean/std/quantiles, mean bias, bias normalized by the test standard deviation, standard-deviation ratio, an approximate pooled W1 distance based on a deterministic bounded subsample, and exact W1 distances between the distributions of per-snapshot spatial means and standard deviations.

## Physical sanity metrics

Velocity is derived from the conservative state as

```text
u = rhou / rho
v = rhov / rho
w = rhow / rho
```

using the convective nondimensionalization already frozen in D3. D13 reports distributions of density mean/std, non-positive density fraction, velocity means/std/RMS values, turbulent kinetic energy, RMS-component isotropy ratio, and non-positive internal-energy fraction, where

```text
rho e = rhoE - 0.5 * (rhou^2 + rhov^2 + rhow^2) / rho.
```

Density or internal-energy violations are reported explicitly rather than clipped away. A small denominator floor is used only to keep derived diagnostics finite when a generated sample is already physically invalid.

## 3-D kinetic-energy spectrum

The HIT mesh contains 33 nodal planes per direction spanning the complete domain and 32 cells per direction. For periodic FFT analysis the maximum-coordinate plane on every axis is therefore removed, leaving the unique `32 x 32 x 32` periodic nodes. This is a post-processing convention only; it does not add periodic graph edges to the DGN.

For each snapshot D13:

1. derives nondimensional velocity;
2. removes the spatial mean of each velocity component;
3. computes a 3-D FFT with `norm="forward"`;
4. forms modal kinetic energy

   ```text
   0.5 * (|u_hat|^2 + |v_hat|^2 + |w_hat|^2);
   ```

5. sums modal energy in spherical radial shells.

Parseval consistency is checked against the real-space fluctuation TKE.

### Nyquist convention

Raw wavenumber is primary. D13 writes both

```text
k * L_ref
k [1/m]
```

and also writes `k/k_Nyquist` as a secondary coordinate.

Only the complete spherical range

```text
0 < |k| <= k_Nyquist,min
```

is used for the radial HIT spectrum. Modes with vector magnitude above that inscribed-sphere limit are FFT corner modes whose individual Cartesian components are still within their axis Nyquist limits; their total energy fraction is reported separately. They are not interpreted as additional physically resolvable wavelengths beyond Nyquist.

`k/k_Nyquist` is used only for compact mesh-relative bands:

```text
low:  0.00 <= k/k_Nyquist < 0.25
mid:  0.25 <= k/k_Nyquist < 0.50
high: 0.50 <= k/k_Nyquist <= 1.00
```

The full spectrum in raw `k` remains the authoritative diagnostic.

## Production outputs

A benchmark run writes

```text
summary.json
channel_metrics.csv
physical_metrics.csv
spectra.csv
spectral_bands.csv
```

`summary.json` records the D2 dataset fingerprint, source generation directory/checkpoint, population sizes, grid convention, Nyquist values, Parseval errors, and explicitly states `generated_reference_pairing: false`.

## D13 validation gate

The Slurm gate uses real held-out HIT snapshots in a self-comparison only to validate the metric implementation. For identical populations it requires zero channel/physical W1 distance and identical spectra, while also checking the real `33^3 -> 32^3` grid convention, `k_Nyquist * L_ref = 32*pi`, density validity, and Parseval consistency.

This identity validation is not a scientific model score. Scientific scores begin only after a trained D12 checkpoint generates an independent population that is compared with the complete frozen D2 test population.
