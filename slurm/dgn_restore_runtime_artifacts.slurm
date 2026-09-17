#!/bin/bash
#SBATCH --job-name=dgn_restore_artifacts
#SBATCH --time=02:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/dgn4avbp/logs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/dgn4avbp/logs/%x_%j.err

set -euo pipefail

PROJECT=/scratch/coop/theret/dgn4avbp
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

cd ${PROJECT}
mkdir -p logs artifacts

echo "========================================"
echo "DGN4AVBP frozen runtime artifact recovery"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"
echo "========================================"

run_py() {
  singularity exec \
    -B /home \
    -B /scratch \
    ${IMAGE} \
    env PYTHONPATH=${PROJECT} \
    ${PYTHON} "$@"
}

echo
echo "STEP 1 - Recreate D1 provenance manifest"
run_py -u ${PROJECT}/scripts/validate_d1_data.py \
  --config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --manifest ${PROJECT}/artifacts/d1_data_manifest.json \
  --quick

echo
echo "STEP 2 - Recreate frozen D2 contiguous split"
run_py -u ${PROJECT}/scripts/create_d2_split.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --split-config ${PROJECT}/configs/data/hit_contiguous_split.yaml \
  --manifest ${PROJECT}/artifacts/d2_split_manifest.json

echo
echo "STEP 3 - Refit frozen D3 preprocessing from D2 train only"
run_py -u ${PROJECT}/scripts/fit_d3_preprocessing.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --split-manifest ${PROJECT}/artifacts/d2_split_manifest.json \
  --reference-config ${PROJECT}/configs/data/hit_reference_scales.yaml \
  --manifest ${PROJECT}/artifacts/d3_preprocessing_manifest.json

echo
echo "STEP 4 - Recreate D4 topology manifest"
run_py -u ${PROJECT}/scripts/validate_d4_topology.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --split-manifest ${PROJECT}/artifacts/d2_split_manifest.json \
  --reference-config ${PROJECT}/configs/data/hit_reference_scales.yaml \
  --preprocessing-manifest ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  --manifest ${PROJECT}/artifacts/d4_topology_manifest.json

echo
echo "STEP 5 - Rebuild D5 hierarchy artifact and manifest"
run_py -u ${PROJECT}/scripts/build_d5_hierarchy.py \
  --data-config ${PROJECT}/configs/data/avbp_hdf5_fixed_mesh_local.yaml \
  --hierarchy-config ${PROJECT}/configs/data/hit_guillard_hierarchy.yaml \
  --split-manifest ${PROJECT}/artifacts/d2_split_manifest.json \
  --reference-config ${PROJECT}/configs/data/hit_reference_scales.yaml \
  --preprocessing-manifest ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  --topology-manifest ${PROJECT}/artifacts/d4_topology_manifest.json \
  --artifact ${PROJECT}/artifacts/d5_hierarchy.pt \
  --manifest ${PROJECT}/artifacts/d5_hierarchy_manifest.json

echo
echo "STEP 6 - Verify recovered artifacts against frozen validated fingerprints"
run_py - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path('/scratch/coop/theret/dgn4avbp/artifacts')
with (root / 'd2_split_manifest.json').open() as f:
    d2 = json.load(f)
with (root / 'd3_preprocessing_manifest.json').open() as f:
    d3 = json.load(f)
with (root / 'd4_topology_manifest.json').open() as f:
    d4 = json.load(f)
with (root / 'd5_hierarchy_manifest.json').open() as f:
    d5 = json.load(f)

expected_dataset = 'e101b858c07e012fcb64ffd731cb4b2d8a41e2f05e3cb6a2b646b28fc89c5d5d'
expected_fit = 'a353fc95aec15c8db4aeab00933c604ae07a3398ec2c53ed7f9a333ef03f021f'
expected_hierarchy = '6750b681f0cc4344ab7520da84d6ca6cf706dc45f1e946a84329b111cb4233dc'
expected_mean = np.array([
    0.9885936305936952,
    -0.004706790750949684,
    0.00057520104191481,
    0.002034316284762378,
    709.527662263822,
], dtype=np.float64)
expected_std = np.array([
    0.0037331995885046773,
    1.1560650350930994,
    1.1460761443451803,
    1.1456123018139601,
    3.41662602439273,
], dtype=np.float64)

assert d2['ordered_file_fingerprint_sha256'] == expected_dataset
assert d2['counts'] == {'train': 1009, 'val': 126, 'test': 126}
assert d3['dataset_fingerprint_sha256'] == expected_dataset
assert d3['fit_file_fingerprint_sha256'] == expected_fit
np.testing.assert_allclose(
    np.asarray(d3['statistical_standardization']['mean'], dtype=np.float64),
    expected_mean,
    rtol=0.0,
    atol=1e-12,
)
np.testing.assert_allclose(
    np.asarray(d3['statistical_standardization']['std'], dtype=np.float64),
    expected_std,
    rtol=0.0,
    atol=1e-12,
)
assert d4['dataset_fingerprint_sha256'] == expected_dataset
assert d4['d3_fit_file_fingerprint_sha256'] == expected_fit
assert d4['topology_contract'] == 'native_hexahedral_connectivity'
assert d4['periodic_closure_added'] is False
assert d5['dataset_fingerprint_sha256'] == expected_dataset
assert d5['d3_fit_file_fingerprint_sha256'] == expected_fit
assert d5['hierarchy_fingerprint_sha256'] == expected_hierarchy
assert d5['num_levels_built'] == 8
assert (root / 'd5_hierarchy.pt').is_file()

print('D2 dataset fingerprint:', d2['ordered_file_fingerprint_sha256'])
print('D3 fit fingerprint:', d3['fit_file_fingerprint_sha256'])
print('D3 mean:', d3['statistical_standardization']['mean'])
print('D3 std:', d3['statistical_standardization']['std'])
print('D4 topology contract:', d4['topology_contract'])
print('D5 hierarchy fingerprint:', d5['hierarchy_fingerprint_sha256'])
print('D5 levels:', d5['num_levels_built'])
print('RECOVERY MATCHES FROZEN D1-D5 CONTRACT')
PY

echo
echo "Recovered runtime files:"
ls -lh \
  ${PROJECT}/artifacts/d1_data_manifest.json \
  ${PROJECT}/artifacts/d2_split_manifest.json \
  ${PROJECT}/artifacts/d3_preprocessing_manifest.json \
  ${PROJECT}/artifacts/d4_topology_manifest.json \
  ${PROJECT}/artifacts/d5_hierarchy.pt \
  ${PROJECT}/artifacts/d5_hierarchy_manifest.json

echo
echo "========================================"
echo "FROZEN RUNTIME ARTIFACT RECOVERY PASSED"
echo "========================================"
