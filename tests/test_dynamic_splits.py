import json

import numpy as np
import pytest

from lc0jax.interpretability.dynamic_splits import (
    root_split_summary,
    split_pair_indices,
    subset_pairs_payload,
)
from tools import split_dynamic_pairs


def test_split_pair_indices_keeps_root_groups_together():
    roots = np.asarray(["root-a", "root-a", "root-b", "root-c", "root-c", "root-d"])

    train, test = split_pair_indices(roots, test_fraction=0.5, seed=7)

    assert set(train.tolist()).isdisjoint(test.tolist())
    assert sorted([*train.tolist(), *test.tolist()]) == list(range(len(roots)))
    train_roots = {roots[idx] for idx in train}
    test_roots = {roots[idx] for idx in test}
    assert train_roots.isdisjoint(test_roots)

    summary = root_split_summary(roots, train, test)
    assert summary["num_rows"] == 6
    assert summary["num_root_groups"] == 4
    assert summary["num_train_rows"] + summary["num_test_rows"] == 6


def test_split_pair_indices_requires_multiple_roots():
    with pytest.raises(ValueError, match="two unique root FENs"):
        split_pair_indices(np.asarray(["same", "same"]), test_fraction=0.5, seed=0)


def test_subset_pairs_payload_preserves_non_row_metadata():
    payload = {
        "differences": np.arange(24).reshape(6, 4),
        "root_fens": np.asarray(["a", "a", "b", "c", "c", "d"], dtype=object),
        "best_moves": np.asarray(["m0", "m1", "m2", "m3", "m4", "m5"], dtype=object),
        "feature_names": np.asarray(["f0", "f1", "f2", "f3"], dtype=object),
        "records_consumed": np.asarray(12, dtype=np.int32),
    }

    subset = subset_pairs_payload(payload, np.asarray([0, 2, 5]), row_count=6)

    np.testing.assert_array_equal(subset["differences"], payload["differences"][[0, 2, 5]])
    assert subset["root_fens"].tolist() == ["a", "b", "d"]
    assert subset["best_moves"].tolist() == ["m0", "m2", "m5"]
    assert subset["feature_names"].tolist() == ["f0", "f1", "f2", "f3"]
    assert int(subset["records_consumed"]) == 12


def test_split_dynamic_pairs_cli_writes_grouped_train_test_files(tmp_path, monkeypatch):
    source = tmp_path / "pairs.npz"
    np.savez_compressed(
        source,
        differences=np.arange(24).reshape(6, 4),
        root_fens=np.asarray(["a", "a", "b", "c", "c", "d"], dtype=object),
        best_moves=np.asarray(["a1a2", "a1a3", "b1b2", "c1c2", "c1c3", "d1d2"], dtype=object),
        subpar_moves=np.asarray(["a1a4", "a1a5", "b1b3", "c1c4", "c1c5", "d1d3"], dtype=object),
        feature_names=np.asarray(["f0", "f1", "f2", "f3"], dtype=object),
        records_consumed=np.asarray(99, dtype=np.int32),
        metadata=np.asarray(
            json.dumps({"activation_key": "token_activations", "num_differences": 6}),
            dtype=object,
        ),
    )
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"

    monkeypatch.setattr(
        "sys.argv",
        [
            "split_dynamic_pairs.py",
            "--pairs",
            str(source),
            "--out-train",
            str(train_path),
            "--out-test",
            str(test_path),
            "--test-fraction",
            "0.5",
            "--seed",
            "3",
        ],
    )

    assert split_dynamic_pairs.main() == 0

    train = np.load(train_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    assert train["differences"].shape[0] + test["differences"].shape[0] == 6
    assert set(train["root_fens"].tolist()).isdisjoint(set(test["root_fens"].tolist()))
    assert train["feature_names"].tolist() == ["f0", "f1", "f2", "f3"]
    assert int(train["records_consumed"]) == 99

    train_metadata = json.loads(str(train["metadata"].item()))
    test_metadata = json.loads(str(test["metadata"].item()))
    assert train_metadata["activation_key"] == "token_activations"
    assert train_metadata["source_num_differences"] == 6
    assert train_metadata["num_differences"] == int(train["differences"].shape[0])
    assert train_metadata["split"]["name"] == "train"
    assert test_metadata["split"]["name"] == "test"
    assert test_metadata["split"]["num_train_rows"] == int(train["differences"].shape[0])
    assert test_metadata["split"]["num_test_rows"] == int(test["differences"].shape[0])
