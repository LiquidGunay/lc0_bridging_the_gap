"""Teachability curriculum artifacts for dynamic concepts."""

from __future__ import annotations

from typing import Any


def _copy_row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"rank"}}


def _curriculum_record(
    row: dict[str, Any],
    *,
    group: str,
    split: str,
    direction_key: str,
    reverse: bool,
) -> dict[str, Any]:
    record = {
        "group": group,
        "split": split,
        "rank": int(row.get("rank", 0)),
        "pair_index": int(row.get("index", -1)),
        "score": float(row.get("score", 0.0)),
        "projection_score": float(row.get("projection_score", row.get("score", 0.0))),
        "direction_key": direction_key,
        "reverse": bool(reverse),
        "root_fen": row.get("root_fens", ""),
        "target_move": row.get("best_moves", ""),
        "contrast_move": row.get("subpar_moves", ""),
        "metadata": _copy_row_metadata(row),
    }
    return record


def teachability_curriculum_records(
    prototypes_report: dict[str, Any],
    *,
    max_prototypes: int | None = None,
    max_controls: int | None = None,
) -> list[dict[str, Any]]:
    """Build JSONL-ready curriculum records from a prototypes report."""
    if max_prototypes is not None and max_prototypes < 0:
        raise ValueError("max_prototypes must be >= 0")
    if max_controls is not None and max_controls < 0:
        raise ValueError("max_controls must be >= 0")

    split = str(prototypes_report.get("split", "train"))
    direction_key = str(prototypes_report.get("direction_key", "direction"))
    reverse = bool(prototypes_report.get("reverse", False))
    prototypes = list(prototypes_report.get("prototypes", []))
    controls = list(prototypes_report.get("random_controls", []))
    if max_prototypes is not None:
        prototypes = prototypes[:max_prototypes]
    if max_controls is not None:
        controls = controls[:max_controls]

    records = [
        _curriculum_record(
            row,
            group="prototype",
            split=split,
            direction_key=direction_key,
            reverse=reverse,
        )
        for row in prototypes
    ]
    records.extend(
        _curriculum_record(
            row,
            group="random_control",
            split=split,
            direction_key=direction_key,
            reverse=reverse,
        )
        for row in controls
    )
    return records
