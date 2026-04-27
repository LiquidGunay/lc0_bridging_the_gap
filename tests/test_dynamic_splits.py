import json

import numpy as np
import pytest

from lc0jax.interpretability.dynamic_splits import (
    root_fen_group_key,
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


def test_split_pair_indices_ignores_fen_fullmove_number():
    same_position_a = "8/8/8/8/8/8/4K3/4k3 w - - 7 1"
    same_position_b = "8/8/8/8/8/8/4K3/4k3 w - - 7 42"
    other_position = "8/8/8/8/8/8/4K3/4k3 b - - 7 1"

    assert root_fen_group_key(same_position_a) == root_fen_group_key(same_position_b)
    train, test = split_pair_indices(
        np.asarray([same_position_a, same_position_b, other_position], dtype=object),
        test_fraction=0.5,
        seed=0,
    )

    train_set = set(train.tolist())
    test_set = set(test.tolist())
    assert {0, 1}.issubset(train_set) or {0, 1}.issubset(test_set)
    assert root_split_summary(
        [same_position_a, same_position_b, other_position],
        train,
        test,
    )["num_root_groups"] == 2


def test_split_pair_indices_requires_multiple_roots():
    with pytest.raises(ValueError, match="two unique root FENs"):
        split_pair_indices(np.asarray(["same", "same"]), test_fraction=0.5, seed=0)


def test_subset_pairs_payload_preserves_non_row_metadata():
    payload = {
        "differences": np.arange(24).reshape(6, 4),
        "root_fens": np.asarray(["a", "a", "b", "c", "c", "d"], dtype=object),
        "best_moves": np.asarray(["m0", "m1", "m2", "m3", "m4", "m5"], dtype=object),
        "feature_names": np.asarray(["f0", "f1", "f2", "f3", "f4", "f5"], dtype=object),
        "sample_weights": np.arange(6),
        "records_consumed": np.asarray(12, dtype=np.int32),
    }

    subset = subset_pairs_payload(payload, np.asarray([0, 2, 5]), row_count=6)

    np.testing.assert_array_equal(subset["differences"], payload["differences"][[0, 2, 5]])
    assert subset["root_fens"].tolist() == ["a", "b", "d"]
    assert subset["best_moves"].tolist() == ["m0", "m2", "m5"]
    assert subset["feature_names"].tolist() == ["f0", "f1", "f2", "f3", "f4", "f5"]
    assert subset["sample_weights"].tolist() == [0, 1, 2, 3, 4, 5]
    assert int(subset["records_consumed"]) == 12

    subset_extra = subset_pairs_payload(
        payload,
        np.asarray([0, 2, 5]),
        row_count=6,
        row_aligned_keys={"sample_weights"},
    )
    assert subset_extra["sample_weights"].tolist() == [0, 2, 5]


def test_split_dynamic_pairs_cli_writes_grouped_train_test_files(tmp_path, monkeypatch):
    source = tmp_path / "pairs.npz"
    np.savez_compressed(
        source,
        differences=np.arange(24).reshape(6, 4),
        root_fens=np.asarray(["a", "a", "b", "c", "c", "d"], dtype=object),
        best_moves=np.asarray(["a1a2", "a1a3", "b1b2", "c1c2", "c1c3", "d1d2"], dtype=object),
        subpar_moves=np.asarray(["a1a4", "a1a5", "b1b3", "c1c4", "c1c5", "d1d3"], dtype=object),
        feature_names=np.asarray(["f0", "f1", "f2", "f3", "f4", "f5"], dtype=object),
        custom_scores=np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int32),
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
            "--row-aligned-key",
            "custom_scores",
        ],
    )

    assert split_dynamic_pairs.main() == 0

    train = np.load(train_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    assert train["differences"].shape[0] + test["differences"].shape[0] == 6
    assert set(train["root_fens"].tolist()).isdisjoint(set(test["root_fens"].tolist()))
    assert train["feature_names"].tolist() == ["f0", "f1", "f2", "f3", "f4", "f5"]
    assert train["custom_scores"].shape[0] == train["differences"].shape[0]
    assert test["custom_scores"].shape[0] == test["differences"].shape[0]
    assert int(train["records_consumed"]) == 99

    train_metadata = json.loads(str(train["metadata"].item()))
    test_metadata = json.loads(str(test["metadata"].item()))
    assert train_metadata["activation_key"] == "token_activations"
    assert train_metadata["source_num_differences"] == 6
    assert train_metadata["num_differences"] == int(train["differences"].shape[0])
    assert train_metadata["split"]["name"] == "train"
    assert train_metadata["split"]["group_key"] == "root_fens_without_fullmove"
    assert train_metadata["split"]["extra_row_aligned_keys"] == ["custom_scores"]
    assert test_metadata["split"]["name"] == "test"
    assert test_metadata["split"]["num_train_rows"] == int(train["differences"].shape[0])
    assert test_metadata["split"]["num_test_rows"] == int(test["differences"].shape[0])
