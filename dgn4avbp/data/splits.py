from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable


_ITERATION_RE = re.compile(r"_(\d+)\.h5$")
_SPLIT_NAMES = ("train", "val", "test")


def load_split_config(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping in split config '{path}'.")
    return cfg


def parse_snapshot_iteration(path: str | Path) -> int:
    name = Path(path).name
    match = _ITERATION_RE.search(name)
    if match is None:
        raise ValueError(
            f"Cannot infer snapshot iteration from '{name}'. Expected a filename ending in _<integer>.h5."
        )
    return int(match.group(1))


def order_files_by_iteration(files: Iterable[str | Path]) -> list[str]:
    ordered = [str(Path(path)) for path in files]
    if not ordered:
        raise ValueError("Cannot split an empty file list.")
    if len(set(ordered)) != len(ordered):
        raise ValueError("Dataset file list contains duplicate paths.")

    decorated = [(parse_snapshot_iteration(path), path) for path in ordered]
    iterations = [iteration for iteration, _ in decorated]
    if len(set(iterations)) != len(iterations):
        raise ValueError("Dataset contains duplicate snapshot iteration identifiers.")

    decorated.sort(key=lambda item: item[0])
    return [path for _, path in decorated]


def file_list_fingerprint(files: Iterable[str | Path]) -> str:
    payload = "\n".join(str(Path(path)) for path in files).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError(f"All split ratios must be > 0, got {ratios}.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Split ratios must sum to 1, got {ratios} with sum={sum(ratios)}.")
    return ratios


def _largest_remainder_counts(num_samples: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if num_samples < len(ratios):
        raise ValueError(
            f"Need at least {len(ratios)} samples to create non-empty train/val/test splits, got {num_samples}."
        )

    quotas = [num_samples * ratio for ratio in ratios]
    counts = [math.floor(quota) for quota in quotas]
    remainder = num_samples - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda idx: (-(quotas[idx] - counts[idx]), idx),
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    if any(count <= 0 for count in counts):
        raise ValueError(
            f"Ratios {ratios} produce an empty split for num_samples={num_samples}: {tuple(counts)}."
        )
    return tuple(counts)  # type: ignore[return-value]


def create_contiguous_split_manifest(
    files: Iterable[str | Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """Create deterministic chronological train/val/test blocks.

    Files are ordered by the integer suffix in their HDF5 filename. No random
    permutation is used. Integer split sizes use the largest-remainder method so
    the requested ratios are approximated as closely as possible while assigning
    every sample exactly once.
    """

    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    ordered_files = order_files_by_iteration(files)
    num_samples = len(ordered_files)
    train_count, val_count, test_count = _largest_remainder_counts(num_samples, ratios)

    train_end = train_count
    val_end = train_count + val_count
    split_files = {
        "train": ordered_files[:train_end],
        "val": ordered_files[train_end:val_end],
        "test": ordered_files[val_end:],
    }

    if len(split_files["test"]) != test_count:
        raise AssertionError("Internal split-count mismatch.")

    iteration_ranges = {}
    for name in _SPLIT_NAMES:
        values = split_files[name]
        iteration_ranges[name] = {
            "first": parse_snapshot_iteration(values[0]),
            "last": parse_snapshot_iteration(values[-1]),
        }

    manifest = {
        "version": 1,
        "strategy": "contiguous_by_snapshot_iteration",
        "rationale": (
            "The HIT files form one time-ordered trajectory; contiguous blocks avoid "
            "randomly interleaving strongly correlated neighboring snapshots across splits."
        ),
        "num_samples": num_samples,
        "ordered_file_fingerprint_sha256": file_list_fingerprint(ordered_files),
        "requested_ratios": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2],
        },
        "counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "actual_ratios": {
            "train": train_count / num_samples,
            "val": val_count / num_samples,
            "test": test_count / num_samples,
        },
        "iteration_ranges": iteration_ranges,
        "boundary_iteration_gaps": {
            "train_to_val": iteration_ranges["val"]["first"] - iteration_ranges["train"]["last"],
            "val_to_test": iteration_ranges["test"]["first"] - iteration_ranges["val"]["last"],
        },
        "files": split_files,
    }
    validate_split_manifest(manifest, ordered_files)
    return manifest


def validate_split_manifest(manifest: dict, dataset_files: Iterable[str | Path]) -> None:
    ordered_files = order_files_by_iteration(dataset_files)
    expected_fingerprint = file_list_fingerprint(ordered_files)
    observed_fingerprint = manifest.get("ordered_file_fingerprint_sha256")
    if observed_fingerprint != expected_fingerprint:
        raise ValueError(
            "Split manifest does not match the current ordered dataset file list: "
            f"expected fingerprint {expected_fingerprint}, got {observed_fingerprint}."
        )

    split_files = manifest.get("files")
    if not isinstance(split_files, dict):
        raise ValueError("Split manifest is missing a 'files' mapping.")

    flattened: list[str] = []
    for name in _SPLIT_NAMES:
        values = split_files.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Split '{name}' must contain a non-empty file list.")
        flattened.extend(str(Path(path)) for path in values)

    if len(flattened) != len(set(flattened)):
        raise ValueError("Split manifest contains file overlap between train/val/test.")
    if flattened != ordered_files:
        raise ValueError(
            "Split manifest does not partition the dataset into three contiguous blocks in chronological order."
        )

    counts = manifest.get("counts", {})
    for name in _SPLIT_NAMES:
        if counts.get(name) != len(split_files[name]):
            raise ValueError(
                f"Split count mismatch for '{name}': manifest says {counts.get(name)}, "
                f"file list contains {len(split_files[name])}."
            )

    if manifest.get("num_samples") != len(ordered_files):
        raise ValueError(
            f"Manifest num_samples={manifest.get('num_samples')} does not match dataset size {len(ordered_files)}."
        )


def split_indices_from_manifest(manifest: dict, dataset_files: Iterable[str | Path]) -> dict[str, list[int]]:
    current_files = [str(Path(path)) for path in dataset_files]
    validate_split_manifest(manifest, current_files)
    index_by_file = {path: index for index, path in enumerate(current_files)}
    return {
        name: [index_by_file[str(Path(path))] for path in manifest["files"][name]]
        for name in _SPLIT_NAMES
    }


def write_split_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_split_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected mapping in split manifest '{path}'.")
    return manifest
