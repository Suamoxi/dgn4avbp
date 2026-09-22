#!/bin/bash
#SBATCH --job-name=dgn_restart500
#SBATCH --time=12:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

CONFIG=${PROJECT}/configs/training/dgn_hit_baseline_c0_30e.yaml
BENCH_CONFIG=${PROJECT}/configs/benchmark/dgn_hit_distribution.yaml
CHECKPOINT=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_best.pt

GEN_ROOT=${PROJECT}/generated/on_manifold_restart/epoch84_t500
BENCH_ROOT=${PROJECT}/benchmark/on_manifold_restart/epoch84_t500
SEAM_ROOT=${PROJECT}/artifacts/on_manifold_restart/epoch84_t500/final_periodic_seams

NUM_SAMPLES=8
BATCH_SIZE=2
SEED=220030
RESTART_T=500

cd ${PROJECT}
mkdir -p logs ${BENCH_ROOT} ${SEAM_ROOT}

echo "========================================"
echo "Epoch-84 paired on-manifold restart test"
echo "========================================"
echo "Git commit: $(git rev-parse HEAD)"
echo "Restart timestep: ${RESTART_T}"
echo "Samples: ${NUM_SAMPLES}"
echo "Same reverse Gaussian noise after t=${RESTART_T}"
echo "========================================"

echo
echo "STEP 1 - Generate free and real-forward-corrupted restart populations"
singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/diagnose_on_manifold_restart.py \
  --config ${CONFIG} \
  --checkpoint ${CHECKPOINT} \
  --restart-t ${RESTART_T} \
  --num-samples ${NUM_SAMPLES} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --trajectory-steps 500 250 100 50 10 0 \
  --output-root ${GEN_ROOT} \
  --overwrite

for MODE in free_x500 real_forward_x500; do
  echo
  echo "STEP 2 - D13 benchmark: ${MODE}"
  singularity exec \
    -B /home \
    -B /scratch \
    ${IMAGE} \
    env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
    ${PYTHON} -u ${PROJECT}/scripts/benchmark_hit_generation.py \
    --config ${BENCH_CONFIG} \
    --generation-dir ${GEN_ROOT}/${MODE} \
    --output-dir ${BENCH_ROOT}/${MODE} \
    --overwrite
done

echo
echo "STEP 3 - Final periodic seam comparison"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/diagnose_periodic_seams.py \
  --config ${CONFIG} \
  --population free_x500=${GEN_ROOT}/free_x500 \
  --population real_forward_x500=${GEN_ROOT}/real_forward_x500 \
  --output-dir ${SEAM_ROOT}

echo
echo "========================================"
echo "ON-MANIFOLD RESTART TEST COMPLETE"
echo "========================================"
echo "Trajectory state: ${GEN_ROOT}/restart_state_trajectory.csv"
echo "Trajectory seams: ${GEN_ROOT}/restart_seam_trajectory.csv"
echo "Trajectory spectrum: ${GEN_ROOT}/restart_spectrum_trajectory.csv"
echo "Free benchmark: ${BENCH_ROOT}/free_x500"
echo "Real-restart benchmark: ${BENCH_ROOT}/real_forward_x500"
echo "Final seam summary: ${SEAM_ROOT}/periodic_seam_summary.csv"
