#!/bin/bash
#SBATCH --job-name=dgn_milestone_diag
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

C0_CONFIG=${PROJECT}/configs/training/dgn_hit_baseline_c0_30e.yaml
C1_CONFIG=${PROJECT}/configs/training/dgn_hit_c1_b8_e100.yaml
BENCH_CONFIG=${PROJECT}/configs/benchmark/dgn_hit_distribution.yaml

GEN_ROOT=${PROJECT}/generated/milestone_diagnostics
BENCH_ROOT=${PROJECT}/benchmark/milestone_diagnostics
OUT_ROOT=${PROJECT}/artifacts/milestone_diagnostics

NUM_SAMPLES=8
GEN_BATCH_SIZE=2
SEED=220030

cd ${PROJECT}
mkdir -p logs ${GEN_ROOT} ${BENCH_ROOT} ${OUT_ROOT}

declare -A CKPT
declare -A CFG

CKPT[22]=${PROJECT}/checkpoints/dgn_hit_baseline_c0_30e_best.pt
CFG[22]=${C0_CONFIG}
CKPT[30]=${PROJECT}/checkpoints/dgn_hit_baseline_c0_30e_last.pt
CFG[30]=${C0_CONFIG}
for E in 40 50 60 70 80 90 100; do
  CKPT[${E}]=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_milestones/epoch_${E}.pt
  CFG[${E}]=${C1_CONFIG}
done
CKPT[84]=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_best.pt
CFG[84]=${C1_CONFIG}

EPOCHS="22 30 40 50 60 70 80 84 90 100"

echo "========================================"
echo "DGN4AVBP milestone diagnostic sweep"
echo "========================================"
echo "Git commit: $(git rev-parse HEAD)"
echo "Epochs: ${EPOCHS}"
echo "Samples/checkpoint: ${NUM_SAMPLES}"
echo "Seed: ${SEED}"
echo "Sampling: full ancestral 1000-step DDPM"
echo "========================================"

for E in ${EPOCHS}; do
  LABEL=epoch${E}
  GENDIR=${GEN_ROOT}/${LABEL}
  BENCHDIR=${BENCH_ROOT}/${LABEL}
  CHECKPOINT=${CKPT[${E}]}
  CONFIG=${CFG[${E}]}

  if [ ! -f "${CHECKPOINT}" ]; then
    echo "Missing checkpoint for epoch ${E}: ${CHECKPOINT}" >&2
    exit 1
  fi

  COUNT=0
  if [ -d "${GENDIR}" ]; then
    COUNT=$(find "${GENDIR}" -maxdepth 1 -name 'sample_*.pt' | wc -l)
  fi

  if [ "${COUNT}" -ne "${NUM_SAMPLES}" ]; then
    echo
    echo "Generating epoch ${E}"
    rm -rf "${GENDIR}"
    singularity exec --nv \
      -B /home \
      -B /scratch \
      ${IMAGE} \
      env PYTHONPATH=${PROJECT} \
      ${PYTHON} -u ${PROJECT}/scripts/generate_hit_dgn.py \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --num-samples ${NUM_SAMPLES} \
      --batch-size ${GEN_BATCH_SIZE} \
      --output-dir "${GENDIR}" \
      --seed ${SEED}
  else
    echo "Reusing ${COUNT} generated samples for epoch ${E}."
  fi

  echo "Benchmarking epoch ${E}"
  singularity exec \
    -B /home \
    -B /scratch \
    ${IMAGE} \
    env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
    ${PYTHON} -u ${PROJECT}/scripts/benchmark_hit_generation.py \
    --config ${BENCH_CONFIG} \
    --generation-dir "${GENDIR}" \
    --output-dir "${BENCHDIR}" \
    --overwrite

done

echo
echo "Aggregating trajectory diagnostics"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
  ${PYTHON} -u ${PROJECT}/scripts/aggregate_milestone_diagnostics.py \
  --training-config ${C0_CONFIG} \
  --generation-root ${GEN_ROOT} \
  --benchmark-root ${BENCH_ROOT} \
  --c0-metrics ${PROJECT}/runs/dgn_hit_baseline_c0_30e/metrics.jsonl \
  --c1-metrics ${PROJECT}/runs/dgn_hit_c1_b8_e100/metrics.jsonl \
  --epochs 22 30 40 50 60 70 80 84 90 100 \
  --output-dir ${OUT_ROOT}

echo
echo "========================================"
echo "MILESTONE DIAGNOSTIC SWEEP COMPLETE"
echo "========================================"
echo "Trajectory: ${OUT_ROOT}/milestone_trajectory.csv"
