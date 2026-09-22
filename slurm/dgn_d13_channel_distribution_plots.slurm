#!/bin/bash
#SBATCH --job-name=dgn_d13_plots
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

GENERATION_DIR=${GENERATION_DIR:-${PROJECT}/generated/c0e22_vs_c1e84/epoch84}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/benchmark/c0e22_vs_c1e84/epoch84}

cd ${PROJECT}
mkdir -p logs

if [ ! -d "${GENERATION_DIR}" ]; then
  echo "Missing generation directory: ${GENERATION_DIR}" >&2
  exit 1
fi

COUNT=$(find "${GENERATION_DIR}" -maxdepth 1 -name 'sample_*.pt' | wc -l)
if [ "${COUNT}" -eq 0 ]; then
  echo "No sample_*.pt files found in ${GENERATION_DIR}" >&2
  exit 1
fi

echo "========================================"
echo "D13 benchmark + channel distribution plots"
echo "========================================"
echo "Generation directory: ${GENERATION_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Generated samples found: ${COUNT}"
echo "Git commit: $(git rev-parse HEAD)"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
  ${PYTHON} -u ${PROJECT}/scripts/benchmark_hit_generation.py \
  --config ${PROJECT}/configs/benchmark/dgn_hit_distribution.yaml \
  --generation-dir ${GENERATION_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --overwrite

echo
echo "========================================"
echo "D13 CSV + DISTRIBUTION PLOTS COMPLETE"
echo "========================================"
echo "Output: ${OUTPUT_DIR}"
