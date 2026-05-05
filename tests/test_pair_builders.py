import json

import numpy as np

from lc0jax.interpretability.pair_builders import (
    iter_rollout_pair_records,
    load_activation_index,
    materialize_rollout_differences,
    trajectory_keys_for_line,
)


def test_materialize_rollout_differences_from_token_activations(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    record = {
        "root_fen": "root",
        "root_history_fens": ["pre", "root"],
        "root_game_id": "game-a",
        "root_game_index": 7,
        "root_ply": 22,
        "root_source": "source.pgn",
        "root_record_id": "record-a",
        "best": {
            "move": "e2e4",
            "score_cp": 50,
            "pv": ["e2e4"],
            "fens": ["root", "best1"],
        },
        "subpar": [
            {
                "move": "d2d4",
                "score_cp": 10,
                "pv": ["d2d4"],
                "fens": ["root", "sub1"],
            }
        ],
    }
    pairs.write_text(json.dumps(record) + "\n", encoding="utf-8")

    shard = tmp_path / "shard_0000.npz"
    token_activations = np.zeros((3, 64, 2), dtype=np.float32)
    token_activations[1, :, 0] = 2.0
    np.savez_compressed(
        shard,
        fens=np.asarray(["root", "best1", "sub1"], dtype=object),
        token_activations=token_activations,
    )

    activation_index, key = load_activation_index(tmp_path)
    assert key == "token_activations"
    payload = materialize_rollout_differences(
        iter_rollout_pair_records(pairs),
        activation_index,
        mode="mean",
    )

    assert payload["differences"].shape == (1, 2)
    np.testing.assert_allclose(payload["differences"][0], [1.0, 0.0])
    assert payload["root_fens"].tolist() == ["root"]
    assert payload["best_moves"].tolist() == ["e2e4"]
    assert payload["subpar_moves"].tolist() == ["d2d4"]
    assert payload["root_history_fens"].tolist() == [["pre", "root"]]
    assert payload["root_game_ids"].tolist() == ["game-a"]
    assert payload["root_game_indices"].tolist() == [7]
    assert payload["root_plies"].tolist() == [22]
    assert payload["root_sources"].tolist() == ["source.pgn"]
    assert payload["root_record_ids"].tolist() == ["record-a"]


def test_load_activation_index_falls_back_to_embeddings(tmp_path):
    shard = tmp_path / "shard_0000.npz"
    np.savez_compressed(
        shard,
        fens=np.asarray(["a", "b"], dtype=object),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    activation_index, key = load_activation_index(tmp_path)
    assert key == "embeddings"
    np.testing.assert_array_equal(activation_index["a"], [1.0, 0.0])


def test_materialize_uses_activation_keys_for_repeated_fens(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    record = {
        "root_fen": "root",
        "best": {
            "move": "e2e4",
            "score_cp": 50,
            "pv": ["e2e4"],
            "fens": ["root", "same"],
            "activation_keys": ["best:0", "best:1"],
        },
        "subpar": [
            {
                "move": "d2d4",
                "score_cp": 10,
                "pv": ["d2d4"],
                "fens": ["root", "same"],
                "activation_keys": ["sub:0", "sub:1"],
            }
        ],
    }
    pairs.write_text(json.dumps(record) + "\n", encoding="utf-8")

    shard = tmp_path / "shard_0000.npz"
    token_activations = np.zeros((4, 64, 2), dtype=np.float32)
    token_activations[1, :, 0] = 2.0
    token_activations[3, :, 1] = 4.0
    np.savez_compressed(
        shard,
        fens=np.asarray(["root", "same", "root", "same"], dtype=object),
        activation_keys=np.asarray(["best:0", "best:1", "sub:0", "sub:1"], dtype=object),
        token_activations=token_activations,
    )

    activation_index, key = load_activation_index(tmp_path)
    assert key == "token_activations"
    assert "same" not in activation_index
    assert trajectory_keys_for_line(record["best"]) == ["best:0", "best:1"]

    payload = materialize_rollout_differences(
        iter_rollout_pair_records(pairs),
        activation_index,
        mode="mean",
    )

    assert payload["differences"].shape == (1, 2)
    np.testing.assert_allclose(payload["differences"][0], [1.0, -2.0])
