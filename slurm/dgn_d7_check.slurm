#!/bin/bash

#SBATCH --job-name=dgn_d7_check
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

cd ${PROJECT}
mkdir -p logs
mkdir -p artifacts

echo "========================================"
echo "DGN4AVBP D7 Improved-DDPM validation"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Project: ${PROJECT}"
echo "Python: ${PYTHON}"
echo "Image: ${IMAGE}"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "========================================"

echo
echo "========================================"
echo "STEP 1 - D1 + D2 + D3 + D4 + D5 + D6 + D7 unit tests"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -m pytest -q \
  ${PROJECT}/tests/test_avbp_hdf5_dataset.py \
  ${PROJECT}/tests/test_splits.py \
  ${PROJECT}/tests/test_preprocessing.py \
  ${PROJECT}/tests/test_topology.py \
  ${PROJECT}/tests/test_hierarchy.py \
  ${PROJECT}/tests/test_dgn_backbone.py \
  ${PROJECT}/tests/test_improved_ddpm.py

echo
echo "========================================"
echo "STEP 2 - Validate canonical Improved-DDPM numerical contract"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} \
  -u ${PROJECT}/scripts/validate_d7_diffusion.py \
  --config ${PROJECT}/configs/diffusion/improved_ddpm_hit.yaml \
  --manifest ${PROJECT}/artifacts/d7_diffusion_manifest.json

echo
echo "========================================"
echo "D7 manifest summary"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import json

path = "/scratch/coop/theret/dgn4avbp/artifacts/d7_diffusion_manifest.json"
with open(path) as f:
    m = json.load(f)

print("contract:", m["contract"])
print("num steps:", m["schedule"]["num_steps"])
print("schedule:", m["schedule"]["type"])
print("beta range:", m["schedule"]["beta_start"], "->", m["schedule"]["beta_end"])
print("alpha_bar final:", m["schedule"]["alpha_bar_final"])
print("lambda_vlb:", m["hybrid_objective"]["lambda_vlb"])
print("sampled VLB multiplier:", m["hybrid_objective"]["sampled_vlb_multiplier"])
print("stop-gradient epsilon/VLB:", m["hybrid_objective"]["stop_gradient_on_epsilon_for_vlb"])
print("t=0 decoder finite:", m["continuous_t0_decoder"]["finite"])
print("t=0 decoder units:", m["continuous_t0_decoder"]["units"])
print("t=0 sampling noise suppressed:", m["sampling"]["noise_suppressed_at_t0"])
print("corrections:")
for item in m["corrections_relative_to_legacy_dgn4cfd_path"]:
    print(" -", item)
PY

echo
echo "========================================"
echo "ALL D0-D7 CHECKS PASSED"
echo "========================================"
