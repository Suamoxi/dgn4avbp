#!/bin/bash
#SBATCH --job-name=dgn_d13_check
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

cd ${PROJECT}
mkdir -p logs artifacts

echo "========================================"
echo "DGN4AVBP D13 3-D HIT distribution benchmark"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "========================================"

echo
echo "========================================"
echo "STEP 1 - D1-D9, D11-D13 unit tests"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -m pytest -q \
  ${PROJECT}/tests/test_avbp_hdf5_dataset.py \
  ${PROJECT}/tests/test_splits.py \
  ${PROJECT}/tests/test_preprocessing.py \
  ${PROJECT}/tests/test_topology.py \
  ${PROJECT}/tests/test_hierarchy.py \
  ${PROJECT}/tests/test_dgn_backbone.py \
  ${PROJECT}/tests/test_improved_ddpm.py \
  ${PROJECT}/tests/test_diffusion_policy.py \
  ${PROJECT}/tests/test_hierarchy_batching.py \
  ${PROJECT}/tests/test_checkpointing.py \
  ${PROJECT}/tests/test_hit_pipeline.py \
  ${PROJECT}/tests/test_hit_benchmark.py

echo
echo "========================================"
echo "STEP 2 - Validate D13 on real held-out HIT snapshots"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/validate_d13_benchmark.py \
  --training-config ${PROJECT}/configs/training/dgn_hit_baseline.yaml \
  --benchmark-config ${PROJECT}/configs/benchmark/dgn_hit_distribution.yaml \
  --num-reference-samples 2 \
  --manifest ${PROJECT}/artifacts/d13_benchmark_manifest.json

echo
echo "========================================"
echo "D13 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json
path = "/scratch/coop/theret/dgn4avbp/artifacts/d13_benchmark_manifest.json"
with open(path) as stream:
    m = json.load(stream)
print("contract:", m["contract"])
print("pairing:", m["generated_reference_pairing"])
print("grid:", m["grid_shape_with_periodic_endpoint"])
print("FFT grid:", m["fft_grid_shape_unique_periodic_nodes"])
print("k_Nyquist * L_ref:", m["k_nyquist_Lref"])
print("k_Nyquist [1/m]:", m["k_nyquist_per_m"])
print("Parseval relative error:", m["reference_max_parseval_relative_error"])
print("corner-mode energy fraction:", m["reference_mean_corner_mode_energy_fraction"])
print("identity checks:", m["identity_checks"])
print("all passed:", m["all_passed"])
PY

echo
echo "========================================"
echo "ALL D0-D9 + D11-D13 CHECKS PASSED (D10 DEFERRED)"
echo "========================================"
