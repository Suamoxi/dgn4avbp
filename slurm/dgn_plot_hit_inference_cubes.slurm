#!/bin/bash
#SBATCH --job-name=dgn_plot_cubes
#SBATCH --time=00:20:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

INPUT_DIR=${INPUT_DIR:-${PROJECT}/generated/c0e22_vs_c1e84/epoch84}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT}/plots/c0e22_vs_c1e84/epoch84}
MAX_SAMPLES=${MAX_SAMPLES:-4}
FIELDS=${FIELDS:-"u_prime v_prime w_prime"}
STATE_SPACE=${STATE_SPACE:-nondimensional}
CMAP=${CMAP:-coolwarm}

cd ${PROJECT}
mkdir -p logs "${OUTPUT_ROOT}"

if [ ! -d "${INPUT_DIR}" ]; then
  echo "Missing inference directory: ${INPUT_DIR}" >&2
  exit 1
fi

COUNT=$(find "${INPUT_DIR}" -maxdepth 1 -name 'sample_*.pt' | wc -l)
if [ "${COUNT}" -eq 0 ]; then
  echo "No sample_*.pt files found in ${INPUT_DIR}" >&2
  exit 1
fi

echo "========================================"
echo "DGN4AVBP inference cube rendering"
echo "========================================"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_ROOT}"
echo "Fields: ${FIELDS}"
echo "Max samples: ${MAX_SAMPLES}"
echo "State space: ${STATE_SPACE}"
echo "Colormap: ${CMAP}"
echo "Git commit: $(git rev-parse HEAD)"
echo "========================================"

for FIELD in ${FIELDS}; do
  echo
  echo "Rendering field: ${FIELD}"
  singularity exec \
    -B /home \
    -B /scratch \
    ${IMAGE} \
    env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
    ${PYTHON} -u ${PROJECT}/scripts/plot_hit_inference_cube.py \
    ${INPUT_DIR} \
    --field ${FIELD} \
    --state-space ${STATE_SPACE} \
    --output-dir ${OUTPUT_ROOT}/${FIELD} \
    --max-samples ${MAX_SAMPLES} \
    --cmap ${CMAP} \
    --robust-percentile 99 \
    --dpi 240 \
    --size 6
done

echo
echo "========================================"
echo "INFERENCE CUBE RENDERING COMPLETE"
echo "========================================"
echo "Plots: ${OUTPUT_ROOT}"
