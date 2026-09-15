#!/bin/bash

#SBATCH --job-name=dgn_d3_check
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
echo "DGN4AVBP D3 preprocessing validation"
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
echo "STEP 1 - D1 + D2 + D3 unit tests"
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
  ${PROJECT}/tests/test_preprocessing.py

echo
echo "========================================"
echo "STEP 2 - Verify frozen D2 manifest against current data"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/create_d2_split.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --split-config ${PROJECT}/configs/data/hit_contiguous_split.yaml \
  --manifest ${PROJECT}/artifacts/d2_split_manifest.json

echo
echo "========================================"
echo "STEP 3 - Fit and validate D3 preprocessing"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/fit_d3_preprocessing.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --split-manifest ${PROJECT}/artifacts/d2_split_manifest.json \
  --reference-config ${PROJECT}/configs/data/hit_reference_scales.yaml \
  --manifest ${PROJECT}/artifacts/d3_preprocessing_manifest.json

echo
echo "========================================"
echo "D3 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json

path = "/scratch/coop/theret/dgn4avbp/artifacts/d3_preprocessing_manifest.json"
with open(path) as f:
    m = json.load(f)

stats = m["statistical_standardization"]
refs = m["physical_nondimensionalization"]["reference_config"]["references"]

print("case_id:", m["case_id"])
print("reference_scheme:", m["reference_scheme"])
print("dataset fingerprint:", m["dataset_fingerprint_sha256"])
print("fit split:", m["fit_split"])
print("fit samples:", m["fit_num_samples"])
print("fit-file fingerprint:", m["fit_file_fingerprint_sha256"])
print("rho_ref:", refs["rho_ref"]["value"])
print("U_ref:", refs["U_ref"]["value"])
print("L_ref:", refs["L_ref"]["value"])
print("T_ref:", refs["T_ref"]["value"])
print("channel means:", stats["mean"])
print("channel stds:", stats["std"])
print("coordinates standardized:", stats["coordinates_standardized"])
PY

echo
echo "========================================"
echo "ALL D0-D3 CHECKS PASSED"
echo "========================================"
