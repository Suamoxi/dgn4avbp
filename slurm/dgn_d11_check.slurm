#!/bin/bash

#SBATCH --job-name=dgn_d11_check
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
echo "DGN4AVBP D11 checkpoint/reproducibility"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Project: ${PROJECT}"
echo "Python: ${PYTHON}"
echo "Image: ${IMAGE}"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "========================================"

echo
echo "========================================"
echo "STEP 1 - D1 through D11 unit tests"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -m pytest -q \
  ${PROJECT}/tests/test_avbp_hdf5_dataset.py \
  ${PROJECT}/tests/test_splits.py \
  ${PROJECT}/tests/test_preprocessing.py \
  ${PROJECT}/tests/test_topology.py \
  ${PROJECT}/tests/test_hierarchy.py \
  ${PROJECT}/tests/test_dgn_backbone.py \
  ${PROJECT}/tests/test_improved_ddpm.py \
  ${PROJECT}/tests/test_diffusion_policy.py \
  ${PROJECT}/tests/test_hierarchy_batching.py \
  ${PROJECT}/tests/test_checkpointing.py

echo
echo "========================================"
echo "STEP 2 - Validate D11 checkpoint state on CUDA"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/validate_d11_checkpointing.py \
  --checkpoint ${PROJECT}/artifacts/d11_probe_checkpoint.pt \
  --manifest ${PROJECT}/artifacts/d11_checkpoint_manifest.json \
  --require-cuda

echo
echo "========================================"
echo "D11 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json

path = "/scratch/coop/theret/dgn4avbp/artifacts/d11_checkpoint_manifest.json"
with open(path) as f:
    m = json.load(f)

print("contract:", m["contract"])
print("device:", m["device"])
print("checkpoint bytes:", m["checkpoint_size_bytes"])
print("model exact:", m["restored"]["model_state_exact"])
print("optimizer exact:", m["restored"]["optimizer_state_exact"])
print("scheduler exact:", m["restored"]["scheduler_state_exact"])
print("AMP scaler exact:", m["restored"]["amp_scaler_state_exact"])
print("step sampler exact:", m["restored"]["loss_second_moment_sampler_state_exact"])
print("progress exact:", m["restored"]["progress_exact"])
print("RNG exact:", m["restored"]["rng"])
print("atomic save:", m["atomic_save"])
print("all passed:", m["all_passed"])
PY

echo
echo "========================================"
echo "ALL D0-D9 + D11 CHECKS PASSED (D10 DEFERRED)"
echo "========================================"
