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


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


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
