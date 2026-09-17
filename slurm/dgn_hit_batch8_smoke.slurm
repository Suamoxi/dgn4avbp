#!/bin/bash
#SBATCH --job-name=dgn_b8_smoke
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
BASE_CONFIG=${PROJECT}/configs/training/dgn_hit_baseline_c0_30e.yaml
SMOKE_CONFIG=${PROJECT}/artifacts/dgn_hit_batch8_smoke.yaml
RUN_DIR=${PROJECT}/runs/dgn_hit_batch8_smoke
CHECKPOINT=${PROJECT}/checkpoints/dgn_hit_batch8_smoke_last.pt
BEST_CHECKPOINT=${PROJECT}/checkpoints/dgn_hit_batch8_smoke_best.pt

cd ${PROJECT}
mkdir -p logs artifacts checkpoints generated runs

echo "========================================"
echo "DGN4AVBP batch-size 8 real HIT memory smoke"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Base config: ${BASE_CONFIG}"
echo "Training batch size: 8"
echo "Training batches: 3"
echo "Validation batches: 1"
echo "========================================"

for required in \
  ${PROJECT}/artifacts/d2_split_manifest.json \
  ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  ${PROJECT}/artifacts/d5_hierarchy.pt \
  ${PROJECT}/artifacts/d5_hierarchy_manifest.json; do
  if [ ! -f "${required}" ]; then
    echo "Missing required runtime artifact: ${required}" >&2
    echo "Run: sbatch slurm/dgn_restore_runtime_artifacts.slurm" >&2
    exit 1
  fi
done

# This job is an isolated disposable smoke test. Remove only its own previous outputs.
rm -rf ${RUN_DIR}
rm -f ${CHECKPOINT} ${BEST_CHECKPOINT} ${SMOKE_CONFIG}
rm -rf ${PROJECT}/generated/dgn_hit_batch8_smoke

nvidia-smi || true

echo
echo "========================================"
echo "STEP 1 - Build isolated batch-8 smoke config"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} - <<'PY'
from pathlib import Path
import yaml

project = Path('/scratch/coop/theret/dgn4avbp')
source = project / 'configs/training/dgn_hit_baseline_c0_30e.yaml'
out = project / 'artifacts/dgn_hit_batch8_smoke.yaml'

with source.open('r', encoding='utf-8') as stream:
    cfg = yaml.safe_load(stream)

cfg['name'] = 'dgn_hit_batch8_smoke'
cfg['training']['epochs'] = 1
cfg['training']['batch_size'] = 8
cfg['training']['checkpoint_path'] = 'checkpoints/dgn_hit_batch8_smoke_last.pt'
cfg['training']['best_checkpoint_path'] = 'checkpoints/dgn_hit_batch8_smoke_best.pt'
cfg['validation']['batch_size'] = 8
cfg['logging']['tensorboard_dir'] = 'runs/dgn_hit_batch8_smoke'
cfg['logging']['metrics_jsonl'] = 'runs/dgn_hit_batch8_smoke/metrics.jsonl'
cfg['generation']['checkpoint_path'] = 'checkpoints/dgn_hit_batch8_smoke_best.pt'
cfg['generation']['output_dir'] = 'generated/dgn_hit_batch8_smoke'
cfg['scientific_contract']['controlled_baseline_probe'] = False
cfg['scientific_contract']['batch8_memory_smoke'] = True
cfg['scientific_contract']['intended_optimizer_steps'] = 3

with out.open('w', encoding='utf-8') as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False)

print(out)
print('training batch size:', cfg['training']['batch_size'])
print('validation batch size:', cfg['validation']['batch_size'])
PY

echo
echo "========================================"
echo "STEP 2 - Three real B=8 training batches + one validation batch"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/train_hit_dgn.py \
  --config ${SMOKE_CONFIG} \
  --epochs 1 \
  --max-train-batches 3 \
  --max-val-batches 1

echo
echo "========================================"
echo "Batch-8 smoke summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json
from pathlib import Path

metrics_path = Path('/scratch/coop/theret/dgn4avbp/runs/dgn_hit_batch8_smoke/metrics.jsonl')
records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
if len(records) != 1:
    raise RuntimeError(f'Expected exactly one smoke epoch record, got {len(records)}')

m = records[0]
peak = int(m['peak_cuda_memory_bytes'])
print('batch size: 8')
print('optimizer steps:', m['global_step'])
print('train weighted loss:', m['train_weighted_loss'])
print('train unweighted loss:', m['train_unweighted_loss'])
print('validation loss:', m['validation_loss'])
print('grad norm:', m['grad_norm'])
print('epoch seconds:', m['epoch_seconds'])
print('peak_cuda_memory_bytes:', peak)
print('peak CUDA allocated GiB:', peak / 1024**3)
print('sampler diagnostics:', m['sampler'])
print('smoke checkpoint:', '/scratch/coop/theret/dgn4avbp/checkpoints/dgn_hit_batch8_smoke_last.pt')
PY

echo
echo "nvidia-smi after smoke process:"
nvidia-smi || true

echo
echo "========================================"
echo "BATCH-8 REAL HIT MEMORY SMOKE PASSED"
echo "========================================"
