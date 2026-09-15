#!/bin/bash

#SBATCH --job-name=dgn_d4_check
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
echo "DGN4AVBP D4 topology validation"
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
echo "STEP 1 - D1 + D2 + D3 + D4 unit tests"
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
  ${PROJECT}/tests/test_topology.py

echo
echo "========================================"
echo "STEP 2 - Validate native HIT topology"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/validate_d4_topology.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --split-manifest ${PROJECT}/artifacts/d2_split_manifest.json \
  --reference-config ${PROJECT}/configs/data/hit_reference_scales.yaml \
  --preprocessing-manifest ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  --manifest ${PROJECT}/artifacts/d4_topology_manifest.json

echo
echo "========================================"
echo "D4 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json

path = "/scratch/coop/theret/dgn4avbp/artifacts/d4_topology_manifest.json"
with open(path) as f:
    m = json.load(f)

grid = m["cartesian_grid"]
geom = m["edge_geometry"]
maps = m["local_hex_edge_mappings"]

print("mesh:", m["mesh_id"])
print("nodes:", m["num_nodes"])
print("cells:", m["num_cells"])
print("directed edges:", m["num_directed_edges"])
print("axis counts:", grid["axis_counts"])
print("physical span:", grid["axis_span"])
print("physical spacing:", grid["axis_spacing"])
print("nondimensional span:", grid["nondimensional_span"])
print("nondimensional spacing:", grid["nondimensional_spacing"])
print("degree histogram:", m["degree_histogram"])
print("nearest-neighbor fraction:", geom["nearest_neighbor_fraction"])
print("cross-box edges:", geom["cross_box_edges"])
print("opposite-face edges:", geom["opposite_face_edges"])
print("local mapping equivalent:", maps["same_undirected_local_edge_set"])
print("global graph equivalent:", maps["same_global_directed_edge_set"])
print("periodic closure added:", m["periodic_closure_added"])
print("D3 geometry consistency:", m["d3_geometry_consistency"])
PY

echo
echo "========================================"
echo "ALL D0-D4 CHECKS PASSED"
echo "========================================"
