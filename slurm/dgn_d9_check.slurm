#!/bin/bash

#SBATCH --job-name=dgn_d9_check
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
echo "DGN4AVBP D9 hierarchy-aware batching"
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
echo "STEP 1 - D1 through D9 unit tests"
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
  ${PROJECT}/tests/test_hierarchy_batching.py

echo
echo "========================================"
echo "STEP 2 - Validate D9 on two real HIT validation snapshots"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/validate_d9_batching.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --split-manifest ${PROJECT}/artifacts/d2_split_manifest.json \
  --reference-config ${PROJECT}/configs/data/hit_reference_scales.yaml \
  --preprocessing-manifest ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  --hierarchy-artifact ${PROJECT}/artifacts/d5_hierarchy.pt \
  --hierarchy-manifest ${PROJECT}/artifacts/d5_hierarchy_manifest.json \
  --model-config ${PROJECT}/configs/model/dgn_hit_baseline.yaml \
  --diffusion-config ${PROJECT}/configs/diffusion/improved_ddpm_hit.yaml \
  --policy-config ${PROJECT}/configs/diffusion/hit_timestep_policy.yaml \
  --manifest ${PROJECT}/artifacts/d9_batching_manifest.json \
  --require-cuda

echo
echo "========================================"
echo "D9 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json

path = "/scratch/coop/theret/dgn4avbp/artifacts/d9_batching_manifest.json"
with open(path) as f:
    m = json.load(f)

fwd = m["forward_validation"]
print("contract:", m["contract"])
print("validated batch size:", m["computational_batch_size_validated"])
print("sample ids:", m["sample_ids"])
print("available hierarchy levels:", m["available_hierarchy_levels"])
print("selected model levels:", m["model_selected_hierarchy_levels"])
print("parent maps preserve graph identity:", m["parent_maps_preserve_graph_identity"])
print("cells preserve graph identity:", m["cell_connectivity_preserves_graph_identity"])
print("deterministic validation batching invariant:", m["deterministic_validation_batching_invariant"])
print("epsilon shape:", fwd["epsilon_shape"])
print("variance shape:", fwd["variance_shape"])
print("batched matches independent:", fwd["batched_matches_independent"])
print("epsilon max abs difference:", fwd["epsilon_max_abs_difference"])
print("variance max abs difference:", fwd["variance_max_abs_difference"])
print("peak CUDA memory bytes:", fwd["peak_cuda_memory_bytes"])
PY

echo
echo "========================================"
echo "ALL D0-D9 CHECKS PASSED"
echo "========================================"
