#!/bin/bash
#SBATCH --job-name=dgn_c1_b8_e100
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
CONFIG=${PROJECT}/configs/training/dgn_hit_c1_b8_e100.yaml

SOURCE_LAST=${PROJECT}/checkpoints/dgn_hit_baseline_c0_30e_last.pt
SOURCE_BEST=${PROJECT}/checkpoints/dgn_hit_baseline_c0_30e_best.pt
RUN_LAST=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_last.pt
RUN_BEST=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_best.pt
ARCHIVE_DIR=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_milestones
METRICS=${PROJECT}/runs/dgn_hit_c1_b8_e100/metrics.jsonl
INIT_RECORD=${PROJECT}/artifacts/c1_b8_e100_initialization.json

cd ${PROJECT}
mkdir -p logs artifacts checkpoints runs ${ARCHIVE_DIR}

checkpoint_epoch() {
  local checkpoint=$1
  singularity exec \
    -B /home \
    -B /scratch \
    ${IMAGE} \
    ${PYTHON} -c "import torch; c=torch.load('${checkpoint}', map_location='cpu', weights_only=False); print(int(c['progress']['epoch']))"
}

echo "========================================"
echo "DGN4AVBP C1: epoch-30 -> epoch-100, batch size 8"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Config: ${CONFIG}"
echo "Source checkpoint: ${SOURCE_LAST}"
echo "Run checkpoint: ${RUN_LAST}"
echo "========================================"

nvidia-smi || true

echo
echo "========================================"
echo "STEP 1 - Validate C0 initialization checkpoint"
echo "========================================"

if [ ! -f "${SOURCE_LAST}" ]; then
  echo "Missing C0 epoch-30 checkpoint: ${SOURCE_LAST}" >&2
  exit 1
fi
if [ ! -f "${SOURCE_BEST}" ]; then
  echo "Missing C0 best checkpoint: ${SOURCE_BEST}" >&2
  exit 1
fi

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - "${SOURCE_LAST}" "${SOURCE_BEST}" "${INIT_RECORD}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import torch

source_last = Path(sys.argv[1])
source_best = Path(sys.argv[2])
record_path = Path(sys.argv[3])

EXPECTED_DATASET = "e101b858c07e012fcb64ffd731cb4b2d8a41e2f05e3cb6a2b646b28fc89c5d5d"
EXPECTED_HIERARCHY = "6750b681f0cc4344ab7520da84d6ca6cf706dc45f1e946a84329b111cb4233dc"

last = torch.load(source_last, map_location="cpu", weights_only=False)
best = torch.load(source_best, map_location="cpu", weights_only=False)

assert int(last["progress"]["epoch"]) == 30, last["progress"]
assert int(last["progress"]["global_step"]) == 15150, last["progress"]
assert int(last["progress"].get("batch_in_epoch", 0)) == 0
assert int(best["progress"]["epoch"]) == 22, best["progress"]

metadata = last.get("metadata", {})
assert metadata.get("dataset_fingerprint_sha256") == EXPECTED_DATASET
assert metadata.get("hierarchy_fingerprint_sha256") == EXPECTED_HIERARCHY

sampler = last["step_sampler"]
counts = np.asarray(sampler["loss_counts"], dtype=np.int64)
assert counts.shape == (1000,)
assert np.all(counts == 10), (int(counts.min()), int(counts.max()))

record = {
    "contract": "c1_batch8_continuation_from_c0_epoch30",
    "source_last": str(source_last),
    "source_best": str(source_best),
    "source_epoch": int(last["progress"]["epoch"]),
    "source_global_step": int(last["progress"]["global_step"]),
    "source_best_epoch": int(best["progress"]["epoch"]),
    "source_best_validation_loss": float(metadata["best_validation_loss"]),
    "dataset_fingerprint_sha256": EXPECTED_DATASET,
    "hierarchy_fingerprint_sha256": EXPECTED_HIERARCHY,
    "sampler_history_min": int(counts.min()),
    "sampler_history_max": int(counts.max()),
    "sampler_fully_warmed": bool(np.all(counts == 10)),
    "batch_size_before": 2,
    "batch_size_after": 8,
    "same_optimization_trajectory_as_c0": False,
}
record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
PY

echo
echo "========================================"
echo "STEP 2 - Initialize isolated C1 state if needed"
echo "========================================"

if [ ! -f "${RUN_LAST}" ]; then
  if [ -f "${METRICS}" ]; then
    echo "Refusing to initialize: metrics already exist but C1 latest checkpoint is missing: ${METRICS}" >&2
    exit 1
  fi
  cp "${SOURCE_LAST}" "${RUN_LAST}"
  cp "${SOURCE_BEST}" "${RUN_BEST}"
  echo "Initialized C1 latest from C0 epoch 30."
  echo "Initialized C1 incumbent best from C0 epoch 22."
else
  echo "Existing C1 checkpoint found; continuing from it."
fi

CURRENT_EPOCH=$(checkpoint_epoch "${RUN_LAST}")
echo "Current C1 epoch: ${CURRENT_EPOCH}"
if [ "${CURRENT_EPOCH}" -lt 30 ] || [ "${CURRENT_EPOCH}" -gt 100 ]; then
  echo "Unexpected C1 checkpoint epoch: ${CURRENT_EPOCH}" >&2
  exit 1
fi

echo
echo "========================================"
echo "STEP 3 - Train in 10-epoch milestones with B=8"
echo "========================================"

for TARGET in 40 50 60 70 80 90 100; do
  CURRENT_EPOCH=$(checkpoint_epoch "${RUN_LAST}")
  ARCHIVE=${ARCHIVE_DIR}/epoch_${TARGET}.pt

  if [ "${CURRENT_EPOCH}" -ge "${TARGET}" ]; then
    echo "Milestone ${TARGET} already reached (current epoch ${CURRENT_EPOCH})."
    if [ ! -f "${ARCHIVE}" ] && [ "${CURRENT_EPOCH}" -eq "${TARGET}" ]; then
      cp "${RUN_LAST}" "${ARCHIVE}"
      echo "Archived existing epoch ${TARGET} checkpoint."
    fi
    continue
  fi

  echo
  echo "----------------------------------------"
  echo "Training from epoch ${CURRENT_EPOCH} to ${TARGET}"
  echo "----------------------------------------"

  singularity exec --nv \
    -B /home \
    -B /scratch \
    ${IMAGE} \
    env PYTHONPATH=${PROJECT} \
    ${PYTHON} -u ${PROJECT}/scripts/train_hit_dgn.py \
    --config ${CONFIG} \
    --resume \
    --epochs ${TARGET}

  REACHED=$(checkpoint_epoch "${RUN_LAST}")
  if [ "${REACHED}" -ne "${TARGET}" ]; then
    echo "Expected epoch ${TARGET}, but latest checkpoint reports epoch ${REACHED}." >&2
    exit 1
  fi

  cp "${RUN_LAST}" "${ARCHIVE}"
  echo "Archived milestone checkpoint: ${ARCHIVE}"
done

echo
echo "========================================"
echo "STEP 4 - C1 summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - "${METRICS}" "${RUN_LAST}" "${RUN_BEST}" "${ARCHIVE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import torch

metrics_path = Path(sys.argv[1])
latest_path = Path(sys.argv[2])
best_path = Path(sys.argv[3])
archive_dir = Path(sys.argv[4])

records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
if not records:
    raise RuntimeError("No C1 metrics were written.")

last = records[-1]
best_record = min(
    (record for record in records if record["validation_loss"] is not None),
    key=lambda record: record["validation_loss"],
)
latest = torch.load(latest_path, map_location="cpu", weights_only=False)
best = torch.load(best_path, map_location="cpu", weights_only=False)

print("C1 epochs recorded:", len(records))
print("latest epoch:", latest["progress"]["epoch"])
print("latest global step:", latest["progress"]["global_step"])
print("latest train weighted loss:", last["train_weighted_loss"])
print("latest train unweighted loss:", last["train_unweighted_loss"])
print("latest validation loss:", last["validation_loss"])
print("best validation within C1 metrics: epoch", best_record["epoch"], "loss", best_record["validation_loss"])
print("incumbent best checkpoint epoch:", best["progress"]["epoch"])
print("incumbent best validation loss:", best["metadata"].get("best_validation_loss"))
print("latest lr:", last["learning_rate"])
print("latest epoch seconds:", last["epoch_seconds"])
print("latest peak CUDA GiB:", last["peak_cuda_memory_bytes"] / 1024**3)
print("sampler diagnostics:", last["sampler"])
print("milestone checkpoints:")
for path in sorted(archive_dir.glob("epoch_*.pt")):
    print(" ", path)
PY

echo
echo "========================================"
echo "C1 BATCH-8 TRAINING THROUGH EPOCH 100 COMPLETE"
echo "========================================"
