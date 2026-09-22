#!/bin/bash
#SBATCH --job-name=dgn_periodic_seam
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
OUT=${PROJECT}/artifacts/periodic_seam_diagnostic

cd ${PROJECT}
mkdir -p logs ${OUT}

echo "========================================"
echo "Periodic endpoint seam diagnostic"
echo "========================================"
echo "Git commit: $(git rev-parse HEAD)"
echo "Reference: all 126 D2 test snapshots"
echo "Generated: epoch84 32-sample baseline + controlled sampler populations"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/diagnose_periodic_seams.py \
  --config ${PROJECT}/configs/training/dgn_hit_baseline_c0_30e.yaml \
  --population epoch84_32=${PROJECT}/generated/c0e22_vs_c1e84/epoch84 \
  --population ancestral8=${PROJECT}/generated/reverse_sampler_comparison/epoch84/ancestral \
  --population vp_sde8=${PROJECT}/generated/reverse_sampler_comparison/epoch84/vp_sde_euler \
  --population vp_ode8=${PROJECT}/generated/reverse_sampler_comparison/epoch84/vp_probability_flow_ode_euler \
  --output-dir ${OUT}

echo
echo "========================================"
echo "PERIODIC SEAM DIAGNOSTIC COMPLETE"
echo "========================================"
echo "Summary: ${OUT}/periodic_seam_summary.csv"
