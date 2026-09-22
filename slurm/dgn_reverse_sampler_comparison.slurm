#!/bin/bash
#SBATCH --job-name=dgn_reverse_cmp
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

GEN_ROOT=${PROJECT}/generated/reverse_sampler_comparison/epoch84
BENCH_ROOT=${PROJECT}/benchmark/reverse_sampler_comparison/epoch84
OUT_ROOT=${PROJECT}/artifacts/reverse_sampler_comparison/epoch84

NUM_SAMPLES=8
BATCH_SIZE=2
SEED=220030

cd ${PROJECT}
mkdir -p logs ${BENCH_ROOT} ${OUT_ROOT}

echo "========================================"
echo "Controlled reverse-sampler comparison"
echo "========================================"
echo "Git commit: $(git rev-parse HEAD)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Samples/method: ${NUM_SAMPLES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Seed: ${SEED}"
echo "Methods:"
echo "  1. Improved-DDPM ancestral"
echo "  2. VP reverse SDE Euler-Maruyama"
echo "  3. VP probability-flow ODE Euler"
echo "========================================"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Missing epoch-84 checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

echo
echo "STEP 0 - Formula/unit tests"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -m pytest -q ${PROJECT}/tests/test_reverse_sampling.py

echo
echo "STEP 1 - Generate controlled populations and reverse trajectories"
singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/compare_reverse_samplers.py \
  --config ${CONFIG} \
  --checkpoint ${CHECKPOINT} \
  --output-root ${GEN_ROOT} \
  --num-samples ${NUM_SAMPLES} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --trajectory-steps 999 900 750 500 250 100 50 10 0 \
  --overwrite

for METHOD in ancestral vp_sde_euler vp_probability_flow_ode_euler; do
  echo
  echo "STEP 2 - D13 benchmark: ${METHOD}"
  singularity exec \
    -B /home \
    -B /scratch \
    ${IMAGE} \
    env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
    ${PYTHON} -u ${PROJECT}/scripts/benchmark_hit_generation.py \
    --config ${BENCH_CONFIG} \
    --generation-dir ${GEN_ROOT}/${METHOD} \
    --output-dir ${BENCH_ROOT}/${METHOD} \
    --overwrite
done

echo
echo "STEP 3 - Aggregate final and trajectory diagnostics"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
  ${PYTHON} -u ${PROJECT}/scripts/aggregate_reverse_sampler_comparison.py \
  --training-config ${CONFIG} \
  --generation-root ${GEN_ROOT} \
  --benchmark-root ${BENCH_ROOT} \
  --output-dir ${OUT_ROOT}

echo
echo "========================================"
echo "REVERSE-SAMPLER COMPARISON COMPLETE"
echo "========================================"
echo "Final comparison:"
echo "  ${OUT_ROOT}/reverse_sampler_comparison.csv"
echo "Trajectory summary:"
echo "  ${OUT_ROOT}/reverse_trajectory_summary.csv"
echo "Metadata:"
echo "  ${GEN_ROOT}/comparison_metadata.json"
