#!/bin/bash
#SBATCH --job-name=dgn_denoise_spec
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
OUT=${PROJECT}/artifacts/denoising_spectrum_diagnostic

cd ${PROJECT}
mkdir -p logs ${OUT}

echo "========================================"
echo "DGN4AVBP denoising spectrum diagnostic"
echo "========================================"
echo "Git commit: $(git rev-parse HEAD)"
echo "Checkpoints: epoch22, epoch84"
echo "Timesteps: 10 50 100 250 500 750 900"
echo "Test samples: 8"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
  ${PYTHON} -u ${PROJECT}/scripts/diagnose_denoising_spectrum.py \
  --training-config ${PROJECT}/configs/training/dgn_hit_baseline_c0_30e.yaml \
  --checkpoint epoch22=${PROJECT}/checkpoints/dgn_hit_baseline_c0_30e_best.pt \
  --checkpoint epoch84=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_best.pt \
  --timesteps 10 50 100 250 500 750 900 \
  --num-test-samples 8 \
  --base-seed 424242 \
  --output-dir ${OUT}

echo
echo "========================================"
echo "DENOISING SPECTRUM DIAGNOSTIC COMPLETE"
echo "========================================"
echo "Output: ${OUT}"
