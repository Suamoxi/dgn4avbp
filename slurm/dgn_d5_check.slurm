#!/bin/bash

#SBATCH --job-name=dgn_d5_check
#SBATCH --time=02:00:00
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
echo "DGN4AVBP D5 fixed hierarchy validation"
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
echo "STEP 1 - D1 + D2 + D3 + D4 + D5 unit tests"
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
  ${PROJECT}/tests/test_hierarchy.py

echo
echo "========================================"
echo "STEP 2 - Build, persist, reload, and validate D5 hierarchy"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/build_d5_hierarchy.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --hierarchy-config ${PROJECT}/configs/data/hit_guillard_hierarchy.yaml \
  --split-manifest ${PROJECT}/artifacts/d2_split_manifest.json \
  --reference-config ${PROJECT}/configs/data/hit_reference_scales.yaml \
  --preprocessing-manifest ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  --topology-manifest ${PROJECT}/artifacts/d4_topology_manifest.json \
  --artifact ${PROJECT}/artifacts/d5_hierarchy.pt \
  --manifest ${PROJECT}/artifacts/d5_hierarchy_manifest.json

echo
echo "========================================"
echo "D5 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json

path = "/scratch/coop/theret/dgn4avbp/artifacts/d5_hierarchy_manifest.json"
with open(path) as f:
    m = json.load(f)

print("algorithm:", m["algorithm"])
print("geometry convention:", m["geometry_convention"])
print("mesh:", m["mesh_id"])
print("levels built:", m["num_levels_built"])
print("stop reason:", m["stop_reason"])
print("max levels requested:", m["max_levels_requested"])
print("max indegree:", m["max_indegree"])
print("periodic closure added:", m["periodic_closure_added"])
print("hierarchy fingerprint:", m["hierarchy_fingerprint_sha256"])
print("model scale selection:", m["model_scale_selection"])
print("levels:")
for level in m["levels"]:
    print(
        "  L{level}: N={nodes} E={edges} edge_nd=[{emin:.6e}, {emean:.6e}, {emax:.6e}]".format(
            level=level["level"],
            nodes=level["num_nodes"],
            edges=level["num_directed_edges"],
            emin=level["min_edge_length_nd"],
            emean=level["mean_edge_length_nd"],
            emax=level["max_edge_length_nd"],
        )
    )
PY

echo
echo "Hierarchy artifact size:"
ls -lh ${PROJECT}/artifacts/d5_hierarchy.pt

echo
echo "========================================"
echo "ALL D0-D5 CHECKS PASSED"
echo "========================================"
