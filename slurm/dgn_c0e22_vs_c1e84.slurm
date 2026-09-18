#!/bin/bash
#SBATCH --job-name=dgn_e22e84
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
C0_CONFIG=${PROJECT}/configs/training/dgn_hit_baseline_c0_30e.yaml\nC1_CONFIG=${PROJECT}/configs/training/dgn_hit_c1_b8_e100.yaml
BENCH_CONFIG=${PROJECT}/configs/benchmark/dgn_hit_distribution.yaml
E22_CKPT=${PROJECT}/checkpoints/dgn_hit_baseline_c0_30e_best.pt
E84_CKPT=${PROJECT}/checkpoints/dgn_hit_c1_b8_e100_best.pt
GEN_ROOT=${PROJECT}/generated/c0e22_vs_c1e84
BENCH_ROOT=${PROJECT}/benchmark/c0e22_vs_c1e84
ARTIFACT=${PROJECT}/artifacts/c0e22_vs_c1e84_comparison.json
NUM_SAMPLES=32
GEN_BATCH_SIZE=2
SEED=220030

cd ${PROJECT}
mkdir -p logs artifacts generated benchmark

# This comparison is intentionally rebuilt from scratch so stale sample_*.pt
# files cannot contaminate either generated population.
rm -rf ${GEN_ROOT} ${BENCH_ROOT}
mkdir -p ${GEN_ROOT}/epoch22 ${GEN_ROOT}/epoch84 ${BENCH_ROOT}

echo "========================================"
echo "DGN4AVBP C0 epoch 22 vs C1 epoch 84 comparison"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Epoch-22 checkpoint: ${E22_CKPT}"
echo "Epoch-84 checkpoint: ${E84_CKPT}"
echo "Generated samples/checkpoint: ${NUM_SAMPLES}"
echo "Generation batch size: ${GEN_BATCH_SIZE}"
echo "Generation seed: ${SEED}"
echo "Sampling: full ancestral 1000-step DDPM"
echo "Reference population: all 126 frozen D2 test snapshots"
echo "========================================"

for required in \
  ${PROJECT}/artifacts/d2_split_manifest.json \
  ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  ${PROJECT}/artifacts/d5_hierarchy.pt \
  ${PROJECT}/artifacts/d5_hierarchy_manifest.json \
  ${E22_CKPT} \
  ${E84_CKPT}; do
  if [ ! -f "${required}" ]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

echo
echo "========================================"
echo "STEP 1 - Verify checkpoint identities and frozen fingerprints"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} - <<'PY'
from pathlib import Path
import torch

project = Path('/scratch/coop/theret/dgn4avbp')
expected_dataset = 'e101b858c07e012fcb64ffd731cb4b2d8a41e2f05e3cb6a2b646b28fc89c5d5d'
expected_hierarchy = '6750b681f0cc4344ab7520da84d6ca6cf706dc45f1e946a84329b111cb4233dc'
checks = [
    ('epoch22', project / 'checkpoints/dgn_hit_baseline_c0_30e_best.pt', 22),
    ('epoch84', project / 'checkpoints/dgn_hit_c1_b8_e100_best.pt', 84),
]
for label, path, expected_epoch in checks:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    progress = ckpt['progress']
    metadata = ckpt.get('metadata', {})
    epoch = int(progress['epoch'])
    if epoch != expected_epoch:
        raise RuntimeError(f'{label}: expected epoch {expected_epoch}, checkpoint contains epoch {epoch}')
    if metadata.get('dataset_fingerprint_sha256') != expected_dataset:
        raise RuntimeError(f'{label}: dataset fingerprint mismatch')
    if metadata.get('hierarchy_fingerprint_sha256') != expected_hierarchy:
        raise RuntimeError(f'{label}: hierarchy fingerprint mismatch')
    print(label, 'epoch=', epoch, 'global_step=', progress['global_step'],
          'best_validation_loss=', metadata.get('best_validation_loss'),
          'role=', metadata.get('checkpoint_role'))
print('checkpoint provenance checks passed')
PY

echo
echo "========================================"
echo "STEP 2 - Generate 32 samples from C0 epoch 22"
echo "========================================"

singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/generate_hit_dgn.py \
  --config ${C0_CONFIG} \
  --checkpoint ${E22_CKPT} \
  --num-samples ${NUM_SAMPLES} \
  --batch-size ${GEN_BATCH_SIZE} \
  --output-dir ${GEN_ROOT}/epoch22 \
  --seed ${SEED}

echo
echo "========================================"
echo "STEP 3 - Generate 32 samples from C1 epoch 84"
echo "========================================"

# The same process seed and sample indices intentionally provide common random
# numbers across checkpoints. D13 nevertheless remains an unpaired population
# benchmark and never treats sample i as a physical target for sample i.
singularity exec --nv \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/generate_hit_dgn.py \
  --config ${C1_CONFIG} \
  --checkpoint ${E84_CKPT} \
  --num-samples ${NUM_SAMPLES} \
  --batch-size ${GEN_BATCH_SIZE} \
  --output-dir ${GEN_ROOT}/epoch84 \
  --seed ${SEED}

echo
echo "========================================"
echo "STEP 4 - D13 benchmark epoch 22 vs all 126 test snapshots"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/benchmark_hit_generation.py \
  --config ${BENCH_CONFIG} \
  --generation-dir ${GEN_ROOT}/epoch22 \
  --output-dir ${BENCH_ROOT}/epoch22

echo
echo "========================================"
echo "STEP 5 - D13 benchmark epoch 84 vs all 126 test snapshots"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  env PYTHONPATH=${PROJECT} \
  ${PYTHON} -u ${PROJECT}/scripts/benchmark_hit_generation.py \
  --config ${BENCH_CONFIG} \
  --generation-dir ${GEN_ROOT}/epoch84 \
  --output-dir ${BENCH_ROOT}/epoch84

echo
echo "========================================"
echo "STEP 6 - Build side-by-side comparison artifact"
echo "========================================"

singularity exec \
  -B /home \
  -B /scratch \
  ${IMAGE} \
  ${PYTHON} - <<'PY'
import csv
import json
from pathlib import Path

project = Path('/scratch/coop/theret/dgn4avbp')
root = project / 'benchmark/c0e22_vs_c1e84'
output = project / 'artifacts/c0e22_vs_c1e84_comparison.json'


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path, key):
    with path.open(newline='') as stream:
        return {row[key]: row for row in csv.DictReader(stream)}


def numeric(row, key):
    return float(row[key])

runs = {}
for label in ('epoch22', 'epoch84'):
    directory = root / label
    summary = read_json(directory / 'summary.json')
    channels = read_csv(directory / 'channel_metrics.csv', 'channel')
    physical = read_csv(directory / 'physical_metrics.csv', 'metric')
    bands = read_csv(directory / 'spectral_bands.csv', 'band')
    runs[label] = {
        'summary': {
            'num_generated_samples': summary['num_generated_samples'],
            'num_reference_samples': summary['num_reference_samples'],
            'source_checkpoints': summary['source_checkpoints'],
            'generated_reference_pairing': summary['generated_reference_pairing'],
            'generated_density_valid_spectra': summary['generated_density_valid_spectra'],
            'generated_max_parseval_relative_error': summary['generated_max_parseval_relative_error'],
            'generated_mean_corner_mode_energy_fraction': summary['generated_mean_corner_mode_energy_fraction'],
        },
        'channels': {
            name: {
                'pooled_wasserstein_1_approx': numeric(row, 'pooled_wasserstein_1_approx'),
                'snapshot_mean_wasserstein_1': numeric(row, 'snapshot_mean_wasserstein_1'),
                'snapshot_std_wasserstein_1': numeric(row, 'snapshot_std_wasserstein_1'),
                'mean_bias_over_reference_std': numeric(row, 'mean_bias_over_reference_std'),
                'std_ratio': numeric(row, 'std_ratio'),
            }
            for name, row in channels.items()
        },
        'physical': {
            name: {
                'generated_mean': numeric(row, 'generated_mean'),
                'reference_mean': numeric(row, 'reference_mean'),
                'mean_bias_over_reference_std': numeric(row, 'mean_bias_over_reference_std'),
                'wasserstein_1': numeric(row, 'wasserstein_1'),
            }
            for name, row in physical.items()
        },
        'spectral_bands': {
            name: {
                'energy_ratio': numeric(row, 'energy_ratio'),
                'absolute_ratio_error': abs(numeric(row, 'energy_ratio') - 1.0),
            }
            for name, row in bands.items()
        },
    }

artifact = {
    'version': 1,
    'contract': 'c0_epoch22_vs_c1_epoch84_d13_unpaired_checkpoint_comparison',
    'generation': {
        'num_samples_per_checkpoint': 32,
        'seed': 220030,
        'same_random_seed_across_checkpoints': True,
        'sampling': 'full_ancestral_1000_step_ddpm',
        'generation_batch_size': 2,
    },
    'reference_population': 'all_126_frozen_D2_test_snapshots',
    'generated_reference_pairing': False,
    'runs': runs,
}
output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')

print('comparison artifact:', output)
for label in ('epoch22', 'epoch84'):
    run = runs[label]
    print('\n', label)
    print('  generated density-valid spectra:', run['summary']['generated_density_valid_spectra'], '/ 32')
    print('  channel pooled W1:')
    for name, values in run['channels'].items():
        print('   ', name, values['pooled_wasserstein_1_approx'])
    print('  physical W1:')
    for name in ('tke', 'isotropy_rms_ratio', 'rho_nonpositive_fraction', 'internal_energy_nonpositive_fraction'):
        if name in run['physical']:
            print('   ', name, run['physical'][name]['wasserstein_1'])
    print('  spectral energy ratios:')
    for name, values in run['spectral_bands'].items():
        print('   ', name, values['energy_ratio'])
PY

echo
echo "========================================"
echo "C0 EPOCH 22 VS C1 EPOCH 84 D13 COMPARISON COMPLETE"
echo "========================================"
echo "Generation root: ${GEN_ROOT}"
echo "Benchmark root: ${BENCH_ROOT}"
echo "Comparison artifact: ${ARTIFACT}"
