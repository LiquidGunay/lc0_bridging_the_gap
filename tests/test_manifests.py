import hashlib
import json

import pytest

from lc0jax.interpretability.manifests import (
    dynamic_roots_manifest,
    file_manifest,
    line_count,
    reference_dataset_manifest,
    sha256_file,
)
from tools import write_reference_manifest


def test_sha256_file_returns_digest_for_existing_file(tmp_path):
    weights = tmp_path / "weights.pb.gz"
    weights.write_bytes(b"weights")

    assert sha256_file(weights) == hashlib.sha256(b"weights").hexdigest()
    assert sha256_file(tmp_path / "missing.pb.gz") is None
    text = tmp_path / "rows.txt"
    text.write_text("a\n\nb\n", encoding="utf-8")
    assert line_count(text) == 2


def test_file_manifest_records_checksum_and_line_count(tmp_path):
    path = tmp_path / "sample.pgn"
    path.write_text("one\n\ntwo\n", encoding="utf-8")

    record = file_manifest(path, role="input", checksum=True, count_lines=True)

    assert record["path"] == str(path)
    assert record["role"] == "input"
    assert record["exists"] is True
    assert record["size_bytes"] == path.stat().st_size
    assert record["sha256"] == hashlib.sha256(b"one\n\ntwo\n").hexdigest()
    assert record["non_empty_lines"] == 2


def test_dynamic_roots_manifest_serializes_paths_and_checksums(tmp_path):
    weights = tmp_path / "weights.pb.gz"
    weights.write_bytes(b"weights")
    roots_path = tmp_path / "roots.records.jsonl"
    outputs_path = tmp_path / "pairs.jsonl"

    manifest = dynamic_roots_manifest(
        run_id="run-a",
        created_utc="2026-05-05T00:00:00+00:00",
        run={"status": "completed", "dry_run": False},
        inputs={"pgn": [tmp_path / "games.pgn"], "fens": []},
        roots={
            "input_mode": "root_records",
            "roots": roots_path,
            "root_count": 10,
            "root_history_complete": True,
            "contains_history_poor_roots": False,
        },
        filters={"max_roots": 10, "min_ply": 12},
        search={"nodes": 800, "multipv": 4},
        model={"weights": weights},
        lc0={"binary": tmp_path / "lc0", "backend": "cuda"},
        outputs={"pairs_jsonl": outputs_path},
        output_status={"pairs_jsonl": {"path": outputs_path, "exists": False}},
    )

    assert manifest["kind"] == "dynamic_roots_v1"
    assert manifest["run"] == {"status": "completed", "dry_run": False}
    assert manifest["root_input_mode"] == "root_records"
    assert manifest["root_history_required"] is True
    assert manifest["root_history_complete"] is True
    assert manifest["contains_history_poor_roots"] is False
    assert manifest["inputs"]["pgn"] == [str(tmp_path / "games.pgn")]
    assert manifest["roots"]["roots"] == str(roots_path)
    assert manifest["filters"]["max_roots"] == 10
    assert manifest["search"]["nodes"] == 800
    assert manifest["model"]["weights"] == str(weights)
    assert manifest["model"]["weights_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert manifest["lc0"]["binary"] == str(tmp_path / "lc0")
    assert manifest["outputs"]["pairs_jsonl"] == str(outputs_path)
    assert manifest["output_status"]["pairs_jsonl"] == {
        "path": str(outputs_path),
        "exists": False,
    }


def test_reference_dataset_manifest_for_human_data(tmp_path):
    source = tmp_path / "human.pgn"
    source.write_text("1. e4 e5\n", encoding="utf-8")

    manifest = reference_dataset_manifest(
        kind="human_reference_v1",
        created_utc="2026-05-09T00:00:00+00:00",
        name="human_reference_v1",
        source={"type": "lichess_standard", "urls": ["https://example.test/human.pgn"]},
        inputs=[file_manifest(source, role="input")],
        filters={"min_elo": 2400, "time_classes": ["rapid", "classical"]},
        dedupe={"key": "board_fen side castling ep"},
        split={"key": "game_id"},
        exclusions=["variant_nonstandard"],
        counts={"games": "10"},
    )

    assert manifest["kind"] == "human_reference_v1"
    assert manifest["name"] == "human_reference_v1"
    assert manifest["source"]["type"] == "lichess_standard"
    assert manifest["inputs"][0]["sha256"] == hashlib.sha256(b"1. e4 e5\n").hexdigest()
    assert manifest["filters"]["min_elo"] == 2400
    assert manifest["dedupe"]["key"] == "board_fen side castling ep"
    assert manifest["split"]["key"] == "game_id"
    assert manifest["exclusions"] == ["variant_nonstandard"]


def test_write_reference_manifest_cli(tmp_path):
    source = tmp_path / "machine.pgn"
    source.write_text("1. d4 d5\n", encoding="utf-8")
    out = tmp_path / "machine_manifest.json"

    assert write_reference_manifest.main(
        [
            "--kind",
            "machine",
            "--name",
            "machine_reference_v1",
            "--source-type",
            "tcec",
            "--source-url",
            "https://example.test/tcec.pgn",
            "--input",
            str(source),
            "--out",
            str(out),
            "--min-ply",
            "18",
            "--min-phase",
            "0.25",
            "--max-phase",
            "0.85",
            "--dedupe-key",
            "board_fen side castling ep",
            "--split-key",
            "game_id",
            "--exclude",
            "tablebase_7_or_less",
            "--count",
            "games=1",
            "--count-lines",
        ]
    ) == 0

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["kind"] == "machine_reference_v1"
    assert manifest["source"]["type"] == "tcec"
    assert manifest["source"]["urls"] == ["https://example.test/tcec.pgn"]
    assert manifest["inputs"][0]["non_empty_lines"] == 1
    assert manifest["filters"]["min_ply"] == 18
    assert manifest["filters"]["min_phase"] == 0.25
    assert manifest["exclusions"] == ["tablebase_7_or_less"]
    assert manifest["counts"] == {"games": 1}


def test_write_reference_manifest_cli_fails_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="allow-missing"):
        write_reference_manifest.main(
            [
                "--kind",
                "human",
                "--name",
                "human_reference_v1",
                "--input",
                str(tmp_path / "missing.pgn"),
                "--out",
                str(tmp_path / "manifest.json"),
            ]
        )


def test_write_reference_manifest_cli_can_allow_missing_file(tmp_path):
    out = tmp_path / "planned_manifest.json"

    assert write_reference_manifest.main(
        [
            "--kind",
            "human",
            "--name",
            "human_reference_v1",
            "--input",
            str(tmp_path / "missing.pgn"),
            "--out",
            str(out),
            "--allow-missing",
        ]
    ) == 0

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["inputs"][0]["exists"] is False
