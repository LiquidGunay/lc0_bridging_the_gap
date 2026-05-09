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
        "root_history_fens": np.asarray(
            [["pa", "a"], ["pa", "a"], ["pb", "b"], ["pc", "c"], ["pc", "c"], ["pd", "d"]],
            dtype=object,
        ),
        "root_game_ids": np.asarray(["ga", "ga", "gb", "gc", "gc", "gd"], dtype=object),
        "best_moves": np.asarray(["m0", "m1", "m2", "m3", "m4", "m5"], dtype=object),
        "score_delta_cp": np.asarray([10, 11, 12, 13, 14, 15], dtype=object),
        "best_wdl": np.asarray(
            [{"wins": idx, "draws": 0, "losses": 0} for idx in range(6)],
            dtype=object,
        ),
        "subpar_wdl": np.asarray(
            [{"wins": 0, "draws": idx, "losses": 0} for idx in range(6)],
            dtype=object,
        ),
        "best_multipv_rank": np.asarray([1, 1, 1, 1, 1, 1], dtype=object),
        "subpar_multipv_rank": np.asarray([2, 3, 2, 4, 2, 3], dtype=object),
        "best_depth": np.asarray([8, 8, 7, 9, 9, 8], dtype=object),
        "subpar_nodes": np.asarray([90, 91, 92, 93, 94, 95], dtype=object),
        "best_raw_info_keys": np.asarray([["depth", "nodes"]] * 6, dtype=object),
        "policy_logits": np.arange(12, dtype=np.float32).reshape(6, 2),
        "feature_names": np.asarray(["f0", "f1", "f2", "f3", "f4", "f5"], dtype=object),
        "sample_weights": np.arange(6),
        "records_consumed": np.asarray(12, dtype=np.int32),
    }

    subset = subset_pairs_payload(payload, np.asarray([0, 2, 5]), row_count=6)

    np.testing.assert_array_equal(subset["differences"], payload["differences"][[0, 2, 5]])
    assert subset["root_fens"].tolist() == ["a", "b", "d"]
    assert subset["root_history_fens"].tolist() == [["pa", "a"], ["pb", "b"], ["pd", "d"]]
    assert subset["root_game_ids"].tolist() == ["ga", "gb", "gd"]
    assert subset["best_moves"].tolist() == ["m0", "m2", "m5"]
    assert subset["score_delta_cp"].tolist() == [10, 12, 15]
    assert subset["best_wdl"].tolist() == [
        {"wins": 0, "draws": 0, "losses": 0},
        {"wins": 2, "draws": 0, "losses": 0},
        {"wins": 5, "draws": 0, "losses": 0},
    ]
    assert subset["subpar_wdl"].tolist() == [
        {"wins": 0, "draws": 0, "losses": 0},
        {"wins": 0, "draws": 2, "losses": 0},
        {"wins": 0, "draws": 5, "losses": 0},
    ]
    assert subset["best_multipv_rank"].tolist() == [1, 1, 1]
    assert subset["subpar_multipv_rank"].tolist() == [2, 2, 3]
    assert subset["best_depth"].tolist() == [8, 7, 8]
    assert subset["subpar_nodes"].tolist() == [90, 92, 95]
    assert subset["best_raw_info_keys"].tolist() == [
        ["depth", "nodes"],
        ["depth", "nodes"],
        ["depth", "nodes"],
    ]
    np.testing.assert_array_equal(subset["policy_logits"], payload["policy_logits"][[0, 2, 5]])
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
        policy_logits=np.arange(12, dtype=np.float32).reshape(6, 2),
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
    assert train["policy_logits"].shape[0] == train["differences"].shape[0]
    assert test["policy_logits"].shape[0] == test["differences"].shape[0]
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
