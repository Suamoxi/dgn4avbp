#!/bin/bash

#SBATCH --job-name=dgn_d6_check
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
echo "DGN4AVBP D6 backbone validation"
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
echo "STEP 1 - D1 + D2 + D3 + D4 + D5 + D6 unit tests"
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
  ${PROJECT}/tests/test_dgn_backbone.py

echo
echo "========================================"
echo "STEP 2 - Validate canonical DGN on real HIT hierarchy"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/validate_d6_backbone.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --split-manifest ${PROJECT}/artifacts/d2_split_manifest.json \
  --reference-config ${PROJECT}/configs/data/hit_reference_scales.yaml \
  --preprocessing-manifest ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  --hierarchy-artifact ${PROJECT}/artifacts/d5_hierarchy.pt \
  --hierarchy-manifest ${PROJECT}/artifacts/d5_hierarchy_manifest.json \
  --model-config ${PROJECT}/configs/model/dgn_hit_baseline.yaml \
  --manifest ${PROJECT}/artifacts/d6_backbone_manifest.json \
  --require-cuda

echo
echo "========================================"
echo "D6 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json

path = "/scratch/coop/theret/dgn4avbp/artifacts/d6_backbone_manifest.json"
with open(path) as f:
    m = json.load(f)

h = m["hierarchy"]
o = m["output_contract"]
f = m["forward_validation"]
p = m["parameter_counts"]

print("model:", m["model_name"])
print("backbone:", m["backbone"])
print("learnable variance:", m["learnable_variance"])
print("available hierarchy levels:", h["available_levels"])
print("selected hierarchy levels:", h["selected_levels"])
print("selected node counts:", [x["num_nodes"] for x in h["selected_level_summaries"]])
print("bottleneck level:", h["bottleneck_level"])
print("bottleneck nodes:", h["bottleneck_nodes"])
print("bottleneck directed edges:", h["bottleneck_directed_edges"])
print("epsilon shape:", o["epsilon_shape"])
print("variance shape:", o["variance_shape"])
print("total output channels:", o["total_channels"])
print("learnable parameters:", p["learnable"])
print("device:", f["device"])
print("peak CUDA memory bytes:", f["peak_cuda_memory_bytes"])
print("fine graph restored:", f["fine_edge_index_restored_after_pool_unpool"])
print("diffusion math status:", m["diffusion_math_status"])
PY

echo
echo "========================================"
echo "ALL D0-D6 CHECKS PASSED"
echo "========================================"
