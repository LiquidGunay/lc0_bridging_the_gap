import hashlib
import json
import sys

import chess
import pytest

from lc0jax.interpretability.mcts_rollouts import RolloutLine, RolloutPairRecord
from tools import build_mcts_pairs


class _FakeEngine:
    def quit(self):
        pass


def test_build_mcts_pairs_fails_fast_by_default(tmp_path, monkeypatch):
    fens = tmp_path / "roots.fens"
    fens.write_text("not-a-fen\n", encoding="utf-8")
    out_jsonl = tmp_path / "pairs.jsonl"

    monkeypatch.setattr(build_mcts_pairs, "_configure_engine", lambda _args: _FakeEngine())
    monkeypatch.setattr(
        build_mcts_pairs,
        "build_rollout_pair_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad fen")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_mcts_pairs.py",
            "--fens",
            str(fens),
            "--out-jsonl",
            str(out_jsonl),
            "--lc0",
            "fake-lc0",
        ],
    )

    with pytest.raises(ValueError, match="bad fen"):
        build_mcts_pairs.main()


def test_build_mcts_pairs_can_skip_expected_position_errors(tmp_path, monkeypatch):
    fens = tmp_path / "roots.fens"
    fens.write_text("not-a-fen\n", encoding="utf-8")
    out_jsonl = tmp_path / "pairs.jsonl"

    monkeypatch.setattr(build_mcts_pairs, "_configure_engine", lambda _args: _FakeEngine())
    monkeypatch.setattr(
        build_mcts_pairs,
        "build_rollout_pair_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad fen")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_mcts_pairs.py",
            "--fens",
            str(fens),
            "--out-jsonl",
            str(out_jsonl),
            "--lc0",
            "fake-lc0",
            "--skip-errors",
        ],
    )

    assert build_mcts_pairs.main() == 0
    assert out_jsonl.read_text(encoding="utf-8") == ""


def test_build_mcts_pairs_root_records_preserve_pre_root_history(tmp_path, monkeypatch):
    board = chess.Board()
    start_fen = board.fen()
    board.push_san("e4")
    root_fen = board.fen()
    best_board = board.copy()
    best_board.push_san("e5")
    subpar_board = board.copy()
    subpar_board.push_san("c5")

    root_records = tmp_path / "roots.records.jsonl"
    root_records.write_text(
        json.dumps(
            {
                "fen": root_fen,
                "history_fens": [start_fen, root_fen],
                "game_id": "game-a",
                "game_index": 3,
                "ply": board.ply(),
                "source": "tiny.pgn",
                "record_id": "record-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_jsonl = tmp_path / "pairs.jsonl"
    out_records = tmp_path / "trajectory.records.jsonl"
    weights = tmp_path / "weights.pb.gz"
    weights.write_bytes(b"weights")

    def fake_build(_engine, fen, _limit, **kwargs):
        assert fen == root_fen
        assert kwargs["search_metadata"]["weights"] == str(weights)
        assert kwargs["search_metadata"]["weights_sha256"] == hashlib.sha256(
            b"weights"
        ).hexdigest()
        assert kwargs["search_metadata"]["multipv"] == 4
        return RolloutPairRecord(
            root_fen=fen,
            node_budget=kwargs["node_budget"],
            best=RolloutLine(
                move="e7e5",
                score_cp=50,
                depth=1,
                nodes=10,
                pv=["e7e5"],
                fens=[root_fen, best_board.fen()],
            ),
            subpar=[
                RolloutLine(
                    move="c7c5",
                    score_cp=10,
                    depth=1,
                    nodes=10,
                    pv=["c7c5"],
                    fens=[root_fen, subpar_board.fen()],
                )
            ],
            root_history_fens=kwargs["root_history_fens"],
            root_game_id=kwargs["root_game_id"],
            root_game_index=kwargs["root_game_index"],
            root_ply=kwargs["root_ply"],
            root_source=kwargs["root_source"],
            root_record_id=kwargs["root_record_id"],
            search_metadata=kwargs["search_metadata"],
        )

    monkeypatch.setattr(build_mcts_pairs, "_configure_engine", lambda _args: _FakeEngine())
    monkeypatch.setattr(build_mcts_pairs, "build_rollout_pair_record", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_mcts_pairs.py",
            "--root-records",
            str(root_records),
            "--out-jsonl",
            str(out_jsonl),
            "--out-trajectory-records",
            str(out_records),
            "--lc0",
            "fake-lc0",
            "--weights",
            str(weights),
            "--history-len",
            "3",
        ],
    )

    assert build_mcts_pairs.main() == 0
    pair = json.loads(out_jsonl.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in out_records.read_text(encoding="utf-8").splitlines()
    ]
    assert pair["root_history_fens"] == [start_fen, root_fen]
    assert pair["root_game_id"] == "game-a"
    assert pair["root_game_index"] == 3
    assert pair["root_ply"] == board.ply()
    assert pair["root_source"] == "tiny.pgn"
    assert pair["root_record_id"] == "record-a"
    assert pair["search"]["weights"] == str(weights)
    assert pair["search"]["weights_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert pair["search"]["nodes"] == 800
    assert pair["search"]["multipv"] == 4
    assert records[0]["history_fens"] == [start_fen, root_fen]
    assert records[1]["history_fens"] == [start_fen, root_fen, best_board.fen()]
    assert records[2]["history_fens"] == [start_fen, root_fen]
    assert records[3]["history_fens"] == [start_fen, root_fen, subpar_board.fen()]
    assert records[0]["root_game_index"] == 3
    assert records[0]["root_ply"] == board.ply()
    assert records[0]["root_record_id"] == "record-a"


def test_root_record_ids_make_activation_line_ids_stable_across_shards():
    root_a = {
        "fen": "root-a",
        "record_id": "source.pgn:game_00000000:ply_0010",
    }
    root_b = {
        "fen": "root-b",
        "record_id": "source.pgn:game_00000001:ply_0010",
    }

    line_a = build_mcts_pairs._line_id_for_root(root_a, source_line=0)
    line_b = build_mcts_pairs._line_id_for_root(root_b, source_line=0)

    assert line_a != line_b
    assert "source.pgn_game_00000000_ply_0010" in line_a
    assert "source.pgn_game_00000001_ply_0010" in line_b
