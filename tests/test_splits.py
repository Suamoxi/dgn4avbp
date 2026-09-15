from __future__ import annotations

from pathlib import Path

import pytest

from dgn4avbp.data.splits import (
    create_contiguous_split_manifest,
    file_list_fingerprint,
    order_files_by_iteration,
    parse_snapshot_iteration,
    split_indices_from_manifest,
    validate_split_manifest,
)


def _files(n: int = 10) -> list[str]:
    return [f"/data/solut_hit_{i:08d}.h5" for i in range(10, 10 * (n + 1), 10)]


def test_parse_snapshot_iteration() -> None:
    assert parse_snapshot_iteration("solut_hit_00013342.h5") == 13342
    with pytest.raises(ValueError):
        parse_snapshot_iteration("last_solution.h5")


def test_order_files_by_iteration_is_numeric() -> None:
    files = [
        "/data/solut_hit_00000100.h5",
        "/data/solut_hit_00000020.h5",
        "/data/solut_hit_00000090.h5",
    ]
    ordered = order_files_by_iteration(files)
    assert [parse_snapshot_iteration(path) for path in ordered] == [20, 90, 100]


def test_contiguous_split_assigns_every_sample_once() -> None:
    files = list(reversed(_files(11)))
    manifest = create_contiguous_split_manifest(files, 0.8, 0.1, 0.1)

    assert manifest["counts"] == {"train": 9, "val": 1, "test": 1}
    assert manifest["files"]["train"] == order_files_by_iteration(files)[:9]
    assert manifest["files"]["val"] == order_files_by_iteration(files)[9:10]
    assert manifest["files"]["test"] == order_files_by_iteration(files)[10:]
    validate_split_manifest(manifest, files)


def test_largest_remainder_counts_for_1261_samples() -> None:
    manifest = create_contiguous_split_manifest(_files(1261), 0.8, 0.1, 0.1)
    assert manifest["counts"] == {"train": 1009, "val": 126, "test": 126}


def test_manifest_rejects_dataset_drift() -> None:
    files = _files(10)
    manifest = create_contiguous_split_manifest(files)
    changed = files[:-1] + ["/data/solut_hit_99999999.h5"]
    with pytest.raises(ValueError, match="fingerprint"):
        validate_split_manifest(manifest, changed)


def test_split_indices_work_when_dataset_file_order_differs() -> None:
    files = _files(10)
    manifest = create_contiguous_split_manifest(files)
    dataset_order = list(reversed(files))
    indices = split_indices_from_manifest(manifest, dataset_order)

    selected_train = [dataset_order[index] for index in indices["train"]]
    assert selected_train == manifest["files"]["train"]


def test_duplicate_iteration_is_rejected() -> None:
    files = [
        "/a/solut_hit_00000010.h5",
        "/b/solut_hit_00000010.h5",
        "/a/solut_hit_00000020.h5",
    ]
    with pytest.raises(ValueError, match="duplicate snapshot iteration"):
        create_contiguous_split_manifest(files)


def test_fingerprint_depends_on_order() -> None:
    files = _files(4)
    assert file_list_fingerprint(files) != file_list_fingerprint(list(reversed(files)))
