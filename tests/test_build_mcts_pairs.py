import sys

import pytest

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
