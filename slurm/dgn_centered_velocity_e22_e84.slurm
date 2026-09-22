#!/bin/bash
#SBATCH --job-name=dgn_centered_vel
#SBATCH --time=00:30:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

E22_DIR=${PROJECT}/generated/c0e22_vs_c1e84/epoch22
E84_DIR=${PROJECT}/generated/c0e22_vs_c1e84/epoch84
OUTPUT=${PROJECT}/artifacts/c0e22_vs_c1e84_centered_velocity.json

cd ${PROJECT}
mkdir -p logs artifacts

for required in \
  ${PROJECT}/artifacts/d2_split_manifest.json \
  ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  ${PROJECT}/artifacts/d5_hierarchy.pt \
  ${PROJECT}/artifacts/d5_hierarchy_manifest.json; do
  if [ ! -f "${required}" ]; then
    echo "Missing required runtime artifact: ${required}" >&2
    exit 1
  fi
done

for directory in "${E22_DIR}" "${E84_DIR}"; do
  if [ ! -d "${directory}" ]; then
    echo "Missing generated population directory: ${directory}" >&2
    exit 1
  fi
  COUNT=$(find "${directory}" -maxdepth 1 -name 'sample_*.pt' | wc -l)
  if [ "${COUNT}" -ne 32 ]; then
    echo "Expected 32 generated samples in ${directory}, found ${COUNT}." >&2
    exit 1
  fi
done

echo "========================================"
echo "DGN4AVBP centered-velocity diagnostic"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Epoch-22 population: ${E22_DIR}"
echo "Epoch-84 population: ${E84_DIR}"
echo "Reference: all 126 frozen D2 test snapshots"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/diagnose_centered_velocity.py \
  --config ${PROJECT}/configs/training/dgn_hit_baseline_c0_30e.yaml \
  --epoch22-dir ${E22_DIR} \
  --epoch84-dir ${E84_DIR} \
  --output ${OUTPUT} \
  --max-pooled-values 500000

echo
echo "========================================"
echo "CENTERED VELOCITY TEST PASSED"
echo "========================================"
echo "Artifact: ${OUTPUT}"
