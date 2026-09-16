#!/bin/bash

#SBATCH --job-name=dgn_d5_order
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
echo "DGN4AVBP D5.1 node-order sensitivity"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/validate_d5_order_sensitivity.py \
  --hierarchy ${PROJECT}/artifacts/d5_hierarchy.pt \
  --output ${PROJECT}/artifacts/d5_order_sensitivity.json \
  --seeds 0 1 2 3

echo
echo "========================================"
echo "D5.1 DIAGNOSTIC COMPLETE"
echo "========================================"
