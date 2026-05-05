import json
import sys

import chess
import chess.pgn
import pytest

from tools import run_dynamic_gpu_pipeline as pipeline


def _fen_after_san(*moves: str) -> str:
    board = chess.Board()
    for move in moves:
        board.push_san(move)
    return board.fen()


def _write_pgn(path, moves: list[str]) -> None:
    game = chess.pgn.Game()
    node = game
    board = game.board()
    for san in moves:
        move = board.parse_san(san)
        node = node.add_variation(move)
        board.push(move)
    path.write_text(str(game) + "\n\n", encoding="utf-8")


def test_resolve_lc0_backend_prefers_gpu_when_jax_gpu_visible():
    parser = pipeline._parser()
    args = parser.parse_args(["--fens", "roots.fens", "--dry-run"])

    assert pipeline.resolve_lc0_backend(args, {"jax_gpu_visible": True}) == "cuda"
    assert pipeline.resolve_lc0_backend(args, {"jax_gpu_visible": False}) is None


def test_parser_accepts_multiple_paths_per_input_flag():
    parser = pipeline._parser()
    args = parser.parse_args(
        [
            "--fens",
            "a.fens",
            "b.fens",
            "--pgn",
            "a.pgn",
            "b.pgn",
            "--dry-run",
        ]
    )

    assert args.fens == ["a.fens", "b.fens"]
    assert args.pgn == ["a.pgn", "b.pgn"]


def test_prepare_roots_filters_dedupes_and_shards(tmp_path):
    fens = [
        _fen_after_san("e4", "e5", "Nf3", "Nc6"),
        _fen_after_san("d4", "d5", "c4", "e6"),
        _fen_after_san("c4", "Nf6", "Nc3", "e5"),
        _fen_after_san("Nf3", "d5", "g3", "c5"),
    ]
    fens_path = tmp_path / "roots.fens"
    fens_path.write_text("\n".join([fens[0], *fens, "not-a-fen"]) + "\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    rc = pipeline.main(
        [
            "--run-dir",
            str(run_dir),
            "--fens",
            str(fens_path),
            "--stop-after",
            "prepare",
            "--min-ply",
            "0",
            "--min-phase",
            "0",
            "--max-phase",
            "1",
            "--min-pieces",
            "2",
            "--min-nonpawn",
            "0",
            "--max-roots",
            "4",
            "--shard-count",
            "2",
            "--shard-index",
            "1",
        ]
    )

    assert rc == 0
    work_dir = run_dir / "shards" / "shard_001_of_002"
    summary = json.loads((work_dir / "run_summary.json").read_text(encoding="utf-8"))
    shard_path = work_dir / "candidate_roots.shard_001_of_002.fens"
    assert summary["roots"]["raw_count"] == 6
    assert summary["roots"]["filtered_count"] == 4
    assert summary["roots"]["root_count"] == 2
    assert summary["run_dir"] == str(run_dir)
    assert summary["work_dir"] == str(work_dir)
    assert summary["roots"]["roots"] == str(shard_path)
    assert shard_path.read_text(encoding="utf-8").splitlines() == [fens[1], fens[3]]


def test_pgn_dry_run_prepares_root_records_by_default(tmp_path):
    pgn_path = tmp_path / "tiny.pgn"
    _write_pgn(pgn_path, ["e4", "e5", "Nf3", "Nc6"])
    run_dir = tmp_path / "run"

    rc = pipeline.main(
        [
            "--run-dir",
            str(run_dir),
            "--pgn",
            str(pgn_path),
            "--dry-run",
            "--stop-after",
            "mcts",
            "--lc0",
            "/fake/lc0",
            "--weights",
            "/fake/weights.pb.gz",
            "--min-ply",
            "0",
            "--min-phase",
            "0",
            "--max-phase",
            "1",
            "--min-pieces",
            "2",
            "--min-nonpawn",
            "0",
            "--max-roots",
            "3",
        ]
    )

    assert rc == 0
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    manifest_path = run_dir / "dynamic_roots_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_path = run_dir / "candidate_roots.filtered.records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    commands = [
        json.loads(line)
        for line in (run_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    mcts_cmd = commands[0]["cmd"]

    assert summary["roots"]["input_mode"] == "root_records"
    assert summary["roots"]["root_arg"] == "--root-records"
    assert summary["roots"]["roots"] == str(records_path)
    assert summary["roots"]["root_count"] == 3
    assert summary["outputs"]["dynamic_roots_manifest"] == str(manifest_path)
    assert manifest["kind"] == "dynamic_roots_v1"
    assert manifest["run"]["status"] == "planned"
    assert manifest["run"]["dry_run"] is True
    assert manifest["root_input_mode"] == "root_records"
    assert manifest["root_history_required"] is True
    assert manifest["root_history_complete"] is True
    assert manifest["contains_history_poor_roots"] is False
    assert manifest["roots"]["source_counts"] == {"fen_input_paths": 0, "pgn_input_paths": 1}
    assert manifest["roots"]["roots"] == str(records_path)
    assert manifest["filters"]["max_roots"] == 3
    assert manifest["search"]["multipv"] == 4
    assert manifest["outputs"]["dynamic_roots_manifest"] == str(manifest_path)
    assert manifest["output_status"]["dynamic_roots_manifest"]["exists"] is True
    assert manifest["output_status"]["pairs_jsonl"]["exists"] is False
    assert len(records) == 3
    assert records[0]["history_fens"][-1] == records[0]["fen"]
    assert len(records[1]["history_fens"]) == 3
    assert "--root-records" in mcts_cmd
    assert str(records_path) in mcts_cmd
    assert "--fens" not in mcts_cmd


def test_mixed_pgn_and_fen_manifest_marks_history_incomplete(tmp_path):
    pgn_path = tmp_path / "tiny.pgn"
    _write_pgn(pgn_path, ["e4", "e5", "Nf3", "Nc6"])
    fens_path = tmp_path / "roots.fens"
    fens_path.write_text(_fen_after_san("d4", "d5", "c4", "e6") + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    rc = pipeline.main(
        [
            "--run-dir",
            str(run_dir),
            "--pgn",
            str(pgn_path),
            "--fens",
            str(fens_path),
            "--dry-run",
            "--stop-after",
            "prepare",
            "--min-ply",
            "0",
            "--min-phase",
            "0",
            "--max-phase",
            "1",
            "--min-pieces",
            "2",
            "--min-nonpawn",
            "0",
        ]
    )

    assert rc == 0
    manifest = json.loads(
        (run_dir / "dynamic_roots_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["root_input_mode"] == "root_records"
    assert manifest["root_history_required"] is True
    assert manifest["root_history_complete"] is False
    assert manifest["contains_history_poor_roots"] is True
    assert manifest["roots"]["history_mode"] == "mixed_pgn_and_root_only_fen"
    assert manifest["roots"]["source_counts"] == {"fen_input_paths": 1, "pgn_input_paths": 1}


def test_dry_run_writes_gpu_command_plan(tmp_path):
    fens_path = tmp_path / "roots.fens"
    fens_path.write_text(_fen_after_san("e4", "e5", "Nf3", "Nc6") + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    rc = pipeline.main(
        [
            "--run-dir",
            str(run_dir),
            "--fens",
            str(fens_path),
            "--dry-run",
            "--lc0",
            "/fake/lc0",
            "--weights",
            "/fake/weights.pb.gz",
            "--max-pairs",
            "2",
            "--min-ply",
            "0",
            "--min-phase",
            "0",
            "--max-phase",
            "1",
            "--min-pieces",
            "2",
            "--min-nonpawn",
            "0",
            "--jax-platform",
            "gpu",
        ]
    )

    assert rc == 0
    records = [
        json.loads(line)
        for line in (run_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["stage"] for record in records] == [
        "mcts",
        "activations",
        "materialize",
        "split",
        "sweep",
    ]
    assert all(record["dry_run"] for record in records)
    assert records[0]["env"]["JAX_PLATFORM_NAME"] == "gpu"
    assert records[0]["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "UV_CACHE_DIR" in records[0]["env"]

    mcts_cmd = records[0]["cmd"]
    assert "--fens" in mcts_cmd
    assert "--root-records" not in mcts_cmd
    if "--backend" in mcts_cmd:
        assert mcts_cmd[mcts_cmd.index("--backend") + 1] == "cuda"


def test_sharded_dry_run_uses_isolated_work_dir(tmp_path):
    fens_path = tmp_path / "roots.fens"
    fens_path.write_text(_fen_after_san("e4", "e5", "Nf3", "Nc6") + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    rc = pipeline.main(
        [
            "--run-dir",
            str(run_dir),
            "--fens",
            str(fens_path),
            "--dry-run",
            "--stop-after",
            "mcts",
            "--lc0",
            "/fake/lc0",
            "--weights",
            "/fake/weights.pb.gz",
            "--min-ply",
            "0",
            "--min-phase",
            "0",
            "--max-phase",
            "1",
            "--min-pieces",
            "2",
            "--min-nonpawn",
            "0",
            "--shard-count",
            "2",
            "--shard-index",
            "0",
        ]
    )

    assert rc == 0
    work_dir = run_dir / "shards" / "shard_000_of_002"
    assert not (run_dir / "commands.jsonl").exists()
    records = [
        json.loads(line)
        for line in (work_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert str(work_dir / "mcts_pairs" / "pairs.jsonl") in records[0]["cmd"]


def test_resume_requires_stage_marker_before_skipping_mcts(tmp_path):
    fens_path = tmp_path / "roots.fens"
    fens_path.write_text(_fen_after_san("e4", "e5", "Nf3", "Nc6") + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    mcts_dir = run_dir / "mcts_pairs"
    mcts_dir.mkdir(parents=True)
    pairs_jsonl = mcts_dir / "pairs.jsonl"
    trajectory_records = mcts_dir / "trajectory.records.jsonl"
    pairs_jsonl.write_text("{}\n", encoding="utf-8")
    trajectory_records.write_text("{}\n", encoding="utf-8")

    rc = pipeline.main(
        [
            "--run-dir",
            str(run_dir),
            "--fens",
            str(fens_path),
            "--dry-run",
            "--resume",
            "--stop-after",
            "mcts",
            "--min-ply",
            "0",
            "--min-phase",
            "0",
            "--max-phase",
            "1",
            "--min-pieces",
            "2",
            "--min-nonpawn",
            "0",
        ]
    )

    assert rc == 0
    records = [
        json.loads(line)
        for line in (run_dir / "commands.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["stage"] for record in records] == ["mcts"]

    run_dir_with_marker = tmp_path / "run_with_marker"
    mcts_dir = run_dir_with_marker / "mcts_pairs"
    mcts_dir.mkdir(parents=True)
    pairs_jsonl = mcts_dir / "pairs.jsonl"
    trajectory_records = mcts_dir / "trajectory.records.jsonl"
    pairs_jsonl.write_text("{}\n", encoding="utf-8")
    trajectory_records.write_text("{}\n", encoding="utf-8")
    pipeline._write_stage_marker(
        pipeline._stage_marker(run_dir_with_marker, "mcts"),
        stage="mcts",
        outputs=[pairs_jsonl, trajectory_records],
    )

    rc = pipeline.main(
        [
            "--run-dir",
            str(run_dir_with_marker),
            "--fens",
            str(fens_path),
            "--dry-run",
            "--resume",
            "--stop-after",
            "mcts",
            "--min-ply",
            "0",
            "--min-phase",
            "0",
            "--max-phase",
            "1",
            "--min-pieces",
            "2",
            "--min-nonpawn",
            "0",
        ]
    )

    assert rc == 0
    assert not (run_dir_with_marker / "commands.jsonl").exists()


def test_non_resume_cleans_stale_activation_dir_before_stage(tmp_path, monkeypatch):
    fens_path = tmp_path / "roots.fens"
    fens_path.write_text(_fen_after_san("e4", "e5", "Nf3", "Nc6") + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    stale_file = run_dir / "activations" / "trajectory_flat" / "stale.npz"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old", encoding="utf-8")
    lc0_path = tmp_path / "lc0"
    weights_path = tmp_path / "weights.pb.gz"
    lc0_path.write_text("#!/bin/sh\n", encoding="utf-8")
    weights_path.write_text("weights", encoding="utf-8")

    def fake_runtime_info():
        return {"jax_gpu_visible": False}

    def fake_run_command(*, stage, cmd, env, log_path, dry_run, marker_path, marker_outputs):
        if stage == "activations":
            assert not stale_file.exists()
        for output in marker_outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("", encoding="utf-8")
        pipeline._write_stage_marker(marker_path, stage=stage, outputs=marker_outputs)

    monkeypatch.setattr(pipeline, "_runtime_info", fake_runtime_info)
    monkeypatch.setattr(pipeline, "_run_command", fake_run_command)

    rc = pipeline.main(
        [
            "--run-dir",
            str(run_dir),
            "--fens",
            str(fens_path),
            "--stop-after",
            "activations",
            "--lc0",
            str(lc0_path),
            "--weights",
            str(weights_path),
            "--min-ply",
            "0",
            "--min-phase",
            "0",
            "--max-phase",
            "1",
            "--min-pieces",
            "2",
            "--min-nonpawn",
            "0",
        ]
    )

    assert rc == 0
    assert (run_dir / "_stage_status" / "activations.done.json").exists()


def test_pipeline_fails_fast_when_filters_drop_all_roots(tmp_path):
    fens_path = tmp_path / "roots.fens"
    fens_path.write_text("not-a-fen\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No candidate root positions"):
        pipeline.main(
            [
                "--run-dir",
                str(tmp_path / "run"),
                "--fens",
                str(fens_path),
                "--dry-run",
            ]
        )


def test_run_command_requires_expected_outputs(tmp_path):
    with pytest.raises(RuntimeError, match="did not create expected outputs"):
        pipeline._run_command(
            stage="noop",
            cmd=[sys.executable, "-c", ""],
            env={},
            log_path=tmp_path / "commands.jsonl",
            dry_run=False,
            marker_path=tmp_path / "_stage_status" / "noop.done.json",
            marker_outputs=[tmp_path / "missing.txt"],
        )
