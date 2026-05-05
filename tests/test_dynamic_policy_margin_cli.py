import sys

import chess
import numpy as np
import pytest

from tools import dynamic_policy_margin


def test_load_pair_rows_skips_illegal_moves_in_cli_validation(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 2), dtype=np.float32),
        root_fens=np.asarray(["8/8/8/8/8/8/8/K6k w - - 0 1"], dtype=object),
        best_moves=np.asarray(["e2e4"], dtype=object),
        subpar_moves=np.asarray(["d2d4"], dtype=object),
    )
    concept = tmp_path / "concept"
    concept.mkdir()
    np.savez_compressed(concept / "concept_direction.npz", direction=np.ones(2))

    monkeypatch.setattr(dynamic_policy_margin, "load_pb_gz", lambda _path: object())
    monkeypatch.setattr(dynamic_policy_margin, "map_bt4_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sys, "argv", [
        "dynamic_policy_margin.py",
        "--pairs",
        str(pairs),
        "--concept",
        str(concept),
        "--pb",
        "fake.pb.gz",
        "--out",
        str(tmp_path / "out.json"),
    ])

    with pytest.raises(ValueError, match="No valid pair rows remained"):
        dynamic_policy_margin.main()


def test_dynamic_policy_margin_uses_root_history_fens(tmp_path, monkeypatch):
    board = chess.Board()
    start_fen = board.fen()
    board.push_san("e4")
    root_fen = board.fen()
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 2), dtype=np.float32),
        root_fens=np.asarray([root_fen], dtype=object),
        root_history_fens=np.asarray([[start_fen, root_fen]], dtype=object),
        best_moves=np.asarray(["e7e5"], dtype=object),
        subpar_moves=np.asarray(["c7c5"], dtype=object),
    )
    concept = tmp_path / "concept"
    concept.mkdir()
    np.savez_compressed(concept / "concept_direction.npz", direction=np.ones(2))
    captured_history = []

    def fake_encode_board(_board, history, **_kwargs):
        captured_history.append([item.fen() for item in history])
        return np.zeros((112, 8, 8), dtype=np.float32)

    def fake_forward(_params, planes, patch=None):
        assert planes.shape == (1, 112, 8, 8)
        return np.zeros((1, 1858), dtype=np.float32), None, None

    monkeypatch.setattr(dynamic_policy_margin, "load_pb_gz", lambda _path: object())
    monkeypatch.setattr(dynamic_policy_margin, "map_bt4_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(dynamic_policy_margin, "encode_board", fake_encode_board)
    monkeypatch.setattr(dynamic_policy_margin, "forward", fake_forward)
    monkeypatch.setattr(
        dynamic_policy_margin,
        "policy_margin_report",
        lambda **kwargs: {"num_pairs": len(kwargs["root_fens"])},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dynamic_policy_margin.py",
            "--pairs",
            str(pairs),
            "--concept",
            str(concept),
            "--pb",
            "fake.pb.gz",
            "--out",
            str(tmp_path / "out.json"),
        ],
    )

    assert dynamic_policy_margin.main() == 0
    assert captured_history == [[start_fen, root_fen]]
