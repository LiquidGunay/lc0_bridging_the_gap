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
            "multipv_rank": 1,
            "depth": 8,
            "nodes": 1000,
            "seldepth": 10,
            "nps": 20000,
            "hashfull": 100,
            "tbhits": 0,
            "wdl": {"wins": 500, "draws": 300, "losses": 200},
            "raw_info_keys": ["depth", "nodes", "wdl"],
            "pv": ["e2e4"],
            "fens": ["root", "best1"],
        },
        "subpar": [
            {
                "move": "d2d4",
                "score_cp": 10,
                "score_delta_cp": 40,
                "multipv_rank": 2,
                "depth": 7,
                "nodes": 900,
                "seldepth": 9,
                "nps": 18000,
                "hashfull": 99,
                "tbhits": 0,
                "wdl": {"wins": 450, "draws": 310, "losses": 240},
                "raw_info_keys": ["depth", "nodes", "wdl"],
                "pv": ["d2d4"],
                "fens": ["root", "sub1"],
            }
        ],
    }
    pairs.write_text(json.dumps(record) + "\n", encoding="utf-8")

    shard = tmp_path / "shard_0000.npz"
    token_activations = np.zeros((3, 64, 2), dtype=np.float32)
    token_activations[1, :, 0] = 2.0
    policy_logits = np.asarray([[0.5, 1.0], [0.0, 0.0], [-0.5, 0.25]], dtype=np.float32)
    np.savez_compressed(
        shard,
        fens=np.asarray(["root", "best1", "sub1"], dtype=object),
        token_activations=token_activations,
        policy_logits=policy_logits,
    )

    activation_index, key = load_activation_index(tmp_path)
    policy_logit_index, policy_key = load_activation_index(
        tmp_path,
        activation_key="policy_logits",
    )
    assert key == "token_activations"
    assert policy_key == "policy_logits"
    payload = materialize_rollout_differences(
        iter_rollout_pair_records(pairs),
        activation_index,
        mode="mean",
        policy_logit_index=policy_logit_index,
    )

    assert payload["differences"].shape == (1, 2)
    np.testing.assert_allclose(payload["differences"][0], [1.0, 0.0])
    np.testing.assert_allclose(payload["policy_logits"], [[0.5, 1.0]])
    assert payload["root_fens"].tolist() == ["root"]
    assert payload["best_moves"].tolist() == ["e2e4"]
    assert payload["subpar_moves"].tolist() == ["d2d4"]
    assert payload["score_delta_cp"].tolist() == [40]
    assert payload["best_wdl"].tolist() == [{"wins": 500, "draws": 300, "losses": 200}]
    assert payload["subpar_wdl"].tolist() == [{"wins": 450, "draws": 310, "losses": 240}]
    assert payload["best_multipv_rank"].tolist() == [1]
    assert payload["subpar_multipv_rank"].tolist() == [2]
    assert payload["best_depth"].tolist() == [8]
    assert payload["subpar_depth"].tolist() == [7]
    assert payload["best_nodes"].tolist() == [1000]
    assert payload["subpar_nodes"].tolist() == [900]
    assert payload["best_seldepth"].tolist() == [10]
    assert payload["subpar_seldepth"].tolist() == [9]
    assert payload["best_nps"].tolist() == [20000]
    assert payload["subpar_nps"].tolist() == [18000]
    assert payload["best_hashfull"].tolist() == [100]
    assert payload["subpar_hashfull"].tolist() == [99]
    assert payload["best_tbhits"].tolist() == [0]
    assert payload["subpar_tbhits"].tolist() == [0]
    assert payload["best_raw_info_keys"].tolist() == [["depth", "nodes", "wdl"]]
    assert payload["subpar_raw_info_keys"].tolist() == [["depth", "nodes", "wdl"]]
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
