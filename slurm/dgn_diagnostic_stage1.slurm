#!/bin/bash
#SBATCH --job-name=dgn_diag_stage1
#SBATCH --time=02:00:00
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
OUT=${PROJECT}/artifacts/diagnostics_stage1

cd ${PROJECT}
mkdir -p logs ${OUT}/normalization ${OUT}/centered_velocity ${OUT}/sampler

echo "========================================"
echo "DGN4AVBP diagnostic stage 1"
echo "========================================"
echo "Git commit: $(git rev-parse HEAD)"
echo "========================================"

echo
echo "STEP 1 - D3 normalization and DC-mode audit"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
  ${PYTHON} -u ${PROJECT}/scripts/diagnose_normalization_dc_modes.py \
  --config ${PROJECT}/configs/training/dgn_hit_baseline_c0_30e.yaml \
  --epoch22-dir ${E22_DIR} \
  --epoch84-dir ${E84_DIR} \
  --output-dir ${OUT}/normalization

echo
echo "STEP 2 - Centered velocity Wasserstein diagnostic"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/diagnose_centered_velocity.py \
  --config ${PROJECT}/configs/training/dgn_hit_baseline_c0_30e.yaml \
  --epoch22-dir ${E22_DIR} \
  --epoch84-dir ${E84_DIR} \
  --output ${OUT}/centered_velocity/centered_velocity.json \
  --max-pooled-values 500000

echo
echo "STEP 3 - Timestep sampler probability audit"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
  ${PYTHON} -u ${PROJECT}/scripts/diagnose_timestep_sampler.py \
  --checkpoint epoch22=${PROJECT}/checkpoints/dgn_hit_baseline_c0_30e_best.pt \
  --checkpoint epoch30=${PROJECT}/checkpoints/dgn_hit_baseline_c0_30e_last.pt \
  --checkpoint epoch40=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_milestones/epoch_40.pt \
  --checkpoint epoch50=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_milestones/epoch_50.pt \
  --checkpoint epoch60=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_milestones/epoch_60.pt \
  --checkpoint epoch70=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_milestones/epoch_70.pt \
  --checkpoint epoch80=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_milestones/epoch_80.pt \
  --checkpoint epoch84=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_best.pt \
  --checkpoint epoch90=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_milestones/epoch_90.pt \
  --checkpoint epoch100=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_milestones/epoch_100.pt \
  --output-dir ${OUT}/sampler

echo
echo "========================================"
echo "DIAGNOSTIC STAGE 1 COMPLETE"
echo "========================================"
echo "Output root: ${OUT}"
