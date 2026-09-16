#!/bin/bash

#SBATCH --job-name=dgn_d8_check
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
mkdir -p logs
mkdir -p artifacts

echo "========================================"
echo "DGN4AVBP D8 timestep/validation policy"
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
echo "STEP 1 - D1 through D8 unit tests"
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
  ${PROJECT}/tests/test_diffusion_policy.py

echo
echo "========================================"
echo "STEP 2 - Validate D8 policy on frozen HIT validation split"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/validate_d8_policy.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --split-manifest ${PROJECT}/artifacts/d2_split_manifest.json \
  --diffusion-config ${PROJECT}/configs/diffusion/improved_ddpm_hit.yaml \
  --policy-config ${PROJECT}/configs/diffusion/hit_timestep_policy.yaml \
  --manifest ${PROJECT}/artifacts/d8_policy_manifest.json \
  --require-cuda

echo
echo "========================================"
echo "D8 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json

path = "/scratch/coop/theret/dgn4avbp/artifacts/d8_policy_manifest.json"
with open(path) as f:
    m = json.load(f)

s = m["training_sampler"]
v = m["validation"]
print("contract:", m["contract"])
print("num diffusion steps:", m["num_diffusion_steps"])
print("training sampler:", s["name"])
print("history per term:", s["history_per_term"])
print("uniform probability:", s["uniform_prob"])
print("importance weight:", s["importance_weight"])
print("CUDA update validated:", s["cuda_update_validated"])
print("validation samples:", v["num_validation_samples"])
print("unique validation timesteps:", v["unique_timesteps"])
print("timestep collisions:", v["timestep_collisions"])
print("timestep range:", v["timestep_min"], "->", v["timestep_max"])
print("fixed across epochs:", v["fixed_across_epochs"])
print("order independent:", v["independent_of_iteration_order"])
print("probe corruption repeat exact:", v["probe_corruption_repeat_exact"])
PY

echo
echo "========================================"
echo "ALL D0-D8 CHECKS PASSED"
echo "========================================"
