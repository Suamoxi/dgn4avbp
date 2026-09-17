#!/bin/bash
#SBATCH --job-name=dgn_c0_30e
#SBATCH --time=16:00:00
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

cd ${PROJECT}
mkdir -p logs artifacts checkpoints generated runs

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_FLAG="--resume"
fi

echo "========================================"
echo "DGN4AVBP controlled physical HIT baseline C0"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Config: ${CONFIG}"
echo "Resume: ${RESUME:-0}"
echo "========================================"

nvidia-smi || true

echo
echo "========================================"
echo "STEP 1 - Frozen-contract regression tests"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -m pytest -q \
  ${PROJECT}/tests/test_avbp_hdf5_dataset.py \
  ${PROJECT}/tests/test_splits.py \
  ${PROJECT}/tests/test_preprocessing.py \
  ${PROJECT}/tests/test_topology.py \
  ${PROJECT}/tests/test_hierarchy.py \
  ${PROJECT}/tests/test_dgn_backbone.py \
  ${PROJECT}/tests/test_improved_ddpm.py \
  ${PROJECT}/tests/test_diffusion_policy.py \
  ${PROJECT}/tests/test_hierarchy_batching.py \
  ${PROJECT}/tests/test_checkpointing.py \
  ${PROJECT}/tests/test_hit_pipeline.py \
  ${PROJECT}/tests/test_hit_benchmark.py

echo
echo "========================================"
echo "STEP 2 - 30-epoch controlled baseline training"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/train_hit_dgn.py \
  --config ${CONFIG} \
  ${RESUME_FLAG}

echo
echo "========================================"
echo "C0 training summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json
from pathlib import Path

metrics_path = Path('/scratch/coop/theret/dgn4avbp/runs/dgn_hit_baseline_c0_30e/metrics.jsonl')
records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
if not records:
    raise RuntimeError('No training metrics were written.')

last = records[-1]
best = min(
    (record for record in records if record['validation_loss'] is not None),
    key=lambda record: record['validation_loss'],
)
print('epochs recorded:', len(records))
print('last epoch:', last['epoch'])
print('last global step:', last['global_step'])
print('last train weighted loss:', last['train_weighted_loss'])
print('last validation loss:', last['validation_loss'])
print('best validation epoch:', best['epoch'])
print('best validation loss:', best['validation_loss'])
print('last lr:', last['learning_rate'])
print('last epoch seconds:', last['epoch_seconds'])
print('last peak CUDA GiB:', last['peak_cuda_memory_bytes'] / 1024**3)
print('sampler diagnostics:', last['sampler'])
print('latest checkpoint:', '/scratch/coop/theret/dgn4avbp/checkpoints/dgn_hit_baseline_c0_30e_last.pt')
print('best checkpoint:', '/scratch/coop/theret/dgn4avbp/checkpoints/dgn_hit_baseline_c0_30e_best.pt')
PY

echo
echo "========================================"
echo "CONTROLLED C0 TRAINING COMPLETE"
echo "========================================"
