"""Teachability curriculum artifacts for dynamic concepts."""

from __future__ import annotations

from typing import Any


def _copy_row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"rank"}}


def _required_row_value(row: dict[str, Any], key: str, *, group: str) -> Any:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"{group} row missing required key '{key}'")
    return value


def _curriculum_record(
    row: dict[str, Any],
    *,
    group: str,
    split: str,
    direction_key: str,
    reverse: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "group": group,
        "split": split,
        "rank": int(_required_row_value(row, "rank", group=group)),
        "pair_index": int(_required_row_value(row, "index", group=group)),
        "score": float(_required_row_value(row, "score", group=group)),
        "projection_score": float(row.get("projection_score", row["score"])),
        "direction_key": direction_key,
        "reverse": bool(reverse),
        "root_fen": _required_row_value(row, "root_fens", group=group),
        "target_move": _required_row_value(row, "best_moves", group=group),
        "contrast_move": _required_row_value(row, "subpar_moves", group=group),
        "provenance": provenance,
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
    provenance = {
        key: prototypes_report[key]
        for key in ("pairs", "concept", "seed", "score_mode")
        if key in prototypes_report
    }
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
            provenance=provenance,
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
            provenance=provenance,
        )
        for row in controls
    )
    return records
