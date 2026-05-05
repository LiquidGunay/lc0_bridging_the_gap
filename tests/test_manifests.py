import hashlib

from lc0jax.interpretability.manifests import dynamic_roots_manifest, sha256_file


def test_sha256_file_returns_digest_for_existing_file(tmp_path):
    weights = tmp_path / "weights.pb.gz"
    weights.write_bytes(b"weights")

    assert sha256_file(weights) == hashlib.sha256(b"weights").hexdigest()
    assert sha256_file(tmp_path / "missing.pb.gz") is None


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
