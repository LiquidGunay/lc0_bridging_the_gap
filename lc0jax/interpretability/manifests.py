"""Reproducibility manifests for dynamic concept datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path | None) -> str | None:
    """Return a SHA256 hex digest for an existing file, otherwise ``None``."""
    if path in (None, ""):
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: str | Path | None) -> int | None:
    """Return non-empty line count for an existing text file, otherwise ``None``."""
    if path in (None, ""):
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def file_manifest(
    path: str | Path,
    *,
    role: str | None = None,
    checksum: bool = True,
    count_lines: bool = False,
) -> dict[str, Any]:
    """Build a small manifest record for one local file path."""
    file_path = Path(path)
    record: dict[str, Any] = {
        "path": file_path,
        "exists": file_path.exists() and file_path.is_file(),
    }
    if role is not None:
        record["role"] = role
    if record["exists"]:
        record["size_bytes"] = int(file_path.stat().st_size)
    if checksum:
        record["sha256"] = sha256_file(file_path)
    if count_lines:
        record["non_empty_lines"] = line_count(file_path)
    return _json_value(record)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def reference_dataset_manifest(
    *,
    kind: str,
    created_utc: str,
    name: str,
    source: dict[str, Any],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]] | None = None,
    filters: dict[str, Any] | None = None,
    dedupe: dict[str, Any] | None = None,
    split: dict[str, Any] | None = None,
    exclusions: list[str] | None = None,
    counts: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build a locked human/machine reference dataset manifest."""
    if kind not in {"human_reference_v1", "machine_reference_v1"}:
        raise ValueError("kind must be human_reference_v1 or machine_reference_v1")
    manifest = {
        "manifest_version": 1,
        "kind": kind,
        "created_utc": created_utc,
        "name": name,
        "source": source,
        "inputs": inputs,
        "outputs": outputs or [],
        "filters": filters or {},
        "dedupe": dedupe or {},
        "split": split or {},
        "exclusions": exclusions or [],
        "counts": counts or {},
    }
    if notes:
        manifest["notes"] = notes
    return _json_value(manifest)


def dynamic_roots_manifest(
    *,
    run_id: str,
    created_utc: str,
    run: dict[str, Any] | None = None,
    inputs: dict[str, Any],
    roots: dict[str, Any],
    filters: dict[str, Any],
    search: dict[str, Any],
    model: dict[str, Any],
    lc0: dict[str, Any],
    outputs: dict[str, Any],
    output_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest for one dynamic-root MCTS/concept run."""
    root_mode = str(roots.get("input_mode", "unknown"))
    root_history_complete = bool(
        roots.get("root_history_complete", root_mode == "root_records")
    )
    contains_history_poor_roots = bool(
        roots.get("contains_history_poor_roots", not root_history_complete)
    )
    manifest = {
        "manifest_version": 1,
        "kind": "dynamic_roots_v1",
        "run_id": run_id,
        "created_utc": created_utc,
        "run": run or {},
        "root_input_mode": root_mode,
        "root_history_required": root_mode == "root_records",
        "root_history_complete": root_history_complete,
        "contains_history_poor_roots": contains_history_poor_roots,
        "inputs": inputs,
        "roots": roots,
        "filters": filters,
        "search": search,
        "model": {
            **model,
            "weights_sha256": sha256_file(model.get("weights")),
        },
        "lc0": lc0,
        "outputs": outputs,
        "output_status": output_status or {},
    }
    return _json_value(manifest)
