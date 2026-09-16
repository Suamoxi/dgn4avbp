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
echo "STEP 2 - Validate native mesh topology"
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

geom = m["edge_geometry"]
maps = m["local_hex_edge_mappings"]
diag = m["optional_diagnostics"]["hit_cartesian"]

print("topology contract:", m["topology_contract"])
print("mesh:", m["mesh_id"])
print("nodes:", m["num_nodes"])
print("cells:", m["num_cells"])
print("directed edges:", m["num_directed_edges"])
print("spatial dim:", m["spatial_dim"])
print("core validation:", m["core_validation"])
print("degree histogram:", m["degree_histogram"]["observed"])
print("generic edge geometry:", geom)
print("local mapping equivalent:", maps["same_undirected_local_edge_set"])
print("global graph equivalent:", maps["same_global_directed_edge_set"])
print("periodic closure added:", m["periodic_closure_added"])
print("D3 geometry consistency:", m["d3_geometry_consistency"])
print("HIT Cartesian diagnostic status:", diag["status"])

if diag["status"] == "passed":
    grid = diag["cartesian_grid"]
    dgeom = diag["edge_geometry"]
    print("diagnostic axis counts:", grid["axis_counts"])
    print("diagnostic physical span:", grid["axis_span"])
    print("diagnostic physical spacing:", grid["axis_spacing"])
    print("diagnostic nondimensional span:", grid["nondimensional_span"])
    print("diagnostic nondimensional spacing:", grid["nondimensional_spacing"])
    print("diagnostic degree histogram:", diag["degree_histogram"])
    print("diagnostic nearest-neighbor fraction:", dgeom["nearest_neighbor_fraction"])
    print("diagnostic cross-box edges:", dgeom["cross_box_edges"])
    print("diagnostic opposite-face edges:", dgeom["opposite_face_edges"])
else:
    print("diagnostic error:", f"{diag['error_type']}: {diag['message']}")
    print("diagnostic failure is non-fatal for the D4 core contract")
PY

echo
echo "========================================"
echo "ALL D0-D4 CORE CHECKS PASSED"
echo "========================================"
