#!/bin/bash

#SBATCH --job-name=dgn_d12_check
#SBATCH --time=01:30:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

cd ${PROJECT}
mkdir -p logs artifacts checkpoints generated runs

echo "========================================"
echo "DGN4AVBP D12 HIT production pipeline"
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
echo "STEP 1 - D1 through D9, D11 and D12 unit tests"
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
  ${PROJECT}/tests/test_checkpointing.py \
  ${PROJECT}/tests/test_hit_pipeline.py

echo
echo "========================================"
echo "STEP 2 - Real HIT train/validation/checkpoint/generation smoke"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/validate_d12_pipeline.py \
  --config ${PROJECT}/configs/training/dgn_hit_baseline.yaml \
  --checkpoint ${PROJECT}/artifacts/d12_smoke_checkpoint.pt \
  --sample ${PROJECT}/artifacts/d12_smoke_generation.pt \
  --manifest ${PROJECT}/artifacts/d12_pipeline_manifest.json \
  --generation-steps 4 \
  --require-cuda

echo
echo "========================================"
echo "D12 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json

path = "/scratch/coop/theret/dgn4avbp/artifacts/d12_pipeline_manifest.json"
with open(path) as f:
    m = json.load(f)

print("contract:", m["contract"])
print("device:", m["device"])
print("split sizes:", m["split_sizes"])
print("training batch size:", m["training"]["batch_size"])
print("training metrics:", m["training"]["metrics"])
print("training peak CUDA memory bytes:", m["training"]["peak_cuda_memory_bytes"])
print("validation deterministic repeat exact:", m["validation"]["deterministic_corruption_repeat_exact"])
print("validation loss:", m["validation"]["loss"])
print("checkpoint model exact after restore:", m["checkpoint"]["model_state_exact_after_restore"])
print("generation steps:", m["generation"]["steps"])
print("generation shape:", m["generation"]["dimensional_shape"])
print("generation finite:", m["generation"]["all_finite"])
print("generation peak CUDA memory bytes:", m["generation"]["peak_cuda_memory_bytes"])
print("all passed:", m["all_passed"])
PY

echo
echo "========================================"
echo "ALL D0-D9 + D11-D12 CHECKS PASSED (D10 DEFERRED)"
echo "========================================"
