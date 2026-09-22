#!/bin/bash
#SBATCH --job-name=dgn_ddim_eta0
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

GEN_DIR=${PROJECT}/generated/ddim_eta0/epoch84
BENCH_DIR=${PROJECT}/benchmark/ddim_eta0/epoch84
SEAM_DIR=${PROJECT}/artifacts/ddim_eta0/epoch84/periodic_seam

NUM_SAMPLES=8
BATCH_SIZE=2
SEED=220030

cd ${PROJECT}
mkdir -p logs ${BENCH_DIR} ${SEAM_DIR}

echo "========================================"
echo "Epoch-84 deterministic DDIM eta=0 test"
echo "========================================"
echo "Git commit: $(git rev-parse HEAD)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Samples: ${NUM_SAMPLES}"
echo "Same initial-noise convention as reverse sampler comparison"
echo "========================================"

echo
echo "STEP 0 - DDIM formula tests"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -m pytest -q ${PROJECT}/tests/test_reverse_sampling.py

echo
echo "STEP 1 - DDIM eta=0 generation"
singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/generate_hit_ddim_eta0.py \
  --config ${CONFIG} \
  --checkpoint ${CHECKPOINT} \
  --output-dir ${GEN_DIR} \
  --num-samples ${NUM_SAMPLES} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --overwrite

echo
echo "STEP 2 - D13 benchmark"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} MPLBACKEND=Agg \
  ${PYTHON} -u ${PROJECT}/scripts/benchmark_hit_generation.py \
  --config ${BENCH_CONFIG} \
  --generation-dir ${GEN_DIR} \
  --output-dir ${BENCH_DIR} \
  --overwrite

echo
echo "STEP 3 - Periodic seam diagnostic"
singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/diagnose_periodic_seams.py \
  --config ${CONFIG} \
  --population ancestral8=${PROJECT}/generated/reverse_sampler_comparison/epoch84/ancestral \
  --population ddim_eta0=${GEN_DIR} \
  --output-dir ${SEAM_DIR}

echo
echo "========================================"
echo "DDIM ETA=0 TEST COMPLETE"
echo "========================================"
echo "D13: ${BENCH_DIR}"
echo "Periodic seams: ${SEAM_DIR}/periodic_seam_summary.csv"
