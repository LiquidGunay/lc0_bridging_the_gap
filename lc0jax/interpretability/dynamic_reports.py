"""Markdown report cards for dynamic rollout concepts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_list(data: np.lib.npyio.NpzFile, key: str) -> list[Any]:
    if key not in data:
        return []
    return [_scalar(item) for item in data[key].tolist()]


def _format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _escape_table(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _list_get(items: list[Any], idx: int, default: Any = "") -> Any:
    return items[idx] if idx < len(items) else default


def _load_pairs_summary(pairs_path: Path) -> dict[str, Any]:
    data = np.load(pairs_path, allow_pickle=True)
    metadata = {}
    if "metadata" in data:
        raw_metadata = _scalar(data["metadata"])
        if raw_metadata:
            metadata = json.loads(str(raw_metadata))
    differences = data["differences"] if "differences" in data else None
    return {
        "metadata": metadata,
        "differences_shape": None if differences is None else tuple(differences.shape),
        "root_fens": _as_list(data, "root_fens"),
        "best_moves": _as_list(data, "best_moves"),
        "subpar_moves": _as_list(data, "subpar_moves"),
        "best_score_cp": _as_list(data, "best_score_cp"),
        "subpar_score_cp": _as_list(data, "subpar_score_cp"),
        "best_pv": _as_list(data, "best_pv"),
        "subpar_pv": _as_list(data, "subpar_pv"),
    }


def _novelty_lines(novelty_report: dict[str, Any] | None) -> list[str]:
    if not novelty_report:
        return ["## Novelty", "", "- report: not provided"]
    lines = [
        "## Novelty",
        "",
        f"- machine samples: {novelty_report.get('machine_samples', 'n/a')}",
        f"- human samples: {novelty_report.get('human_samples', 'n/a')}",
        f"- accepted vectors: {novelty_report.get('accepted_vectors', [])}",
    ]
    for vector in novelty_report.get("vectors", []):
        lines.append(
            "- vector {vector}: novelty_area={area}, positive_rank_fraction={fraction}".format(
                vector=vector.get("vector", "n/a"),
                area=_format_float(vector.get("novelty_area")),
                fraction=_format_float(vector.get("positive_rank_fraction")),
            )
        )
    return lines


def _baseline_lines(baselines_report: dict[str, Any] | None) -> list[str]:
    if not baselines_report:
        return ["## Baselines", "", "- report: not provided"]
    actual = baselines_report.get("actual", {})
    random_sparse = baselines_report.get("random_sparse", {})
    shuffled_labels = baselines_report.get("shuffled_labels", {})
    shuffled_solve = baselines_report.get("shuffled_solve", {})
    return [
        "## Baselines",
        "",
        f"- nonzero features: {baselines_report.get('nonzero_features', 'n/a')}",
        (
            "- actual constraint satisfaction: "
            f"{_format_float(actual.get('constraint_satisfaction'))}"
        ),
        f"- actual mean score: {_format_float(actual.get('mean_score'))}",
        (
            "- random sparse constraint satisfaction mean: "
            f"{_format_float(random_sparse.get('constraint_satisfaction_mean'))}"
        ),
        (
            "- shuffled-label constraint satisfaction mean: "
            f"{_format_float(shuffled_labels.get('constraint_satisfaction_mean'))}"
        ),
        (
            "- shuffled-solve count: "
            f"{shuffled_solve.get('count', 0)}"
        ),
    ]


def _evaluation_lines(evaluation_report: dict[str, Any] | None) -> list[str]:
    if not evaluation_report:
        return ["## Held-Out Evaluation", "", "- report: not provided"]
    metrics = evaluation_report.get("evaluation", {})
    return [
        "## Held-Out Evaluation",
        "",
        f"- split: {evaluation_report.get('split', 'n/a')}",
        f"- samples: {evaluation_report.get('num_pairs', 'n/a')}",
        f"- dimension: {evaluation_report.get('dimension', 'n/a')}",
        f"- direction key: {evaluation_report.get('direction_key', 'n/a')}",
        f"- nonzero features: {evaluation_report.get('nonzero_features', 'n/a')}",
        (
            "- constraint satisfaction: "
            f"{_format_float(metrics.get('constraint_satisfaction'))}"
        ),
        f"- margin satisfaction: {_format_float(metrics.get('margin_satisfaction'))}",
        f"- mean score: {_format_float(metrics.get('mean_score'))}",
        f"- min score: {_format_float(metrics.get('min_score'))}",
    ]


def _policy_margin_lines(policy_margin_report: dict[str, Any] | None) -> list[str]:
    if not policy_margin_report:
        return ["## Policy-Margin Patch", "", "- report: not provided"]
    return [
        "## Policy-Margin Patch",
        "",
        f"- samples: {policy_margin_report.get('num_pairs', 'n/a')}",
        f"- layer: {policy_margin_report.get('layer', 'n/a')}",
        f"- alpha: {policy_margin_report.get('alpha', 'n/a')}",
        f"- mean base margin: {_format_float(policy_margin_report.get('mean_base_margin'))}",
        (
            "- mean patched margin: "
            f"{_format_float(policy_margin_report.get('mean_patched_margin'))}"
        ),
        (
            "- mean delta margin: "
            f"{_format_float(policy_margin_report.get('mean_delta_margin'))}"
        ),
        (
            "- fraction delta positive: "
            f"{_format_float(policy_margin_report.get('fraction_delta_positive'))}"
        ),
        f"- top1 change rate: {_format_float(policy_margin_report.get('top1_change_rate'))}",
        f"- top1 legal masked: {policy_margin_report.get('top1_legal_masked', 'n/a')}",
    ]


def build_dynamic_concept_report(
    *,
    pairs_path: str | Path,
    concept_dir: str | Path,
    novelty_path: str | Path | None = None,
    evaluation_path: str | Path | None = None,
    baselines_path: str | Path | None = None,
    policy_margin_path: str | Path | None = None,
    top_n: int = 10,
) -> str:
    """Return a markdown report for a dynamic concept run."""
    pairs_path = Path(pairs_path)
    concept_dir = Path(concept_dir)
    solver_report = _read_json(concept_dir / "report.json")
    pairs = _load_pairs_summary(pairs_path)

    if novelty_path is None:
        candidate = concept_dir / "novelty_report.json"
        novelty_report = _read_json(candidate) if candidate.exists() else None
    else:
        novelty_report = _read_json(Path(novelty_path))

    if baselines_path is None:
        candidate = concept_dir / "baselines_report.json"
        baselines_report = _read_json(candidate) if candidate.exists() else None
    else:
        baselines_report = _read_json(Path(baselines_path))

    if evaluation_path is None:
        candidate = concept_dir / "heldout_eval_report.json"
        evaluation_report = _read_json(candidate) if candidate.exists() else None
    else:
        evaluation_report = _read_json(Path(evaluation_path))

    if policy_margin_path is None:
        candidate = concept_dir / "policy_margin_report.json"
        policy_margin_report = _read_json(candidate) if candidate.exists() else None
    else:
        policy_margin_report = _read_json(Path(policy_margin_path))

    metadata = pairs["metadata"]
    differences_shape = pairs["differences_shape"]
    lines = [
        "# Dynamic Concept Report",
        "",
        "## Solver",
        "",
        f"- concept dir: `{concept_dir}`",
        f"- solver pairs: `{solver_report.get('pairs', 'n/a')}`",
        f"- report pairs: `{pairs_path}`",
        f"- method: {solver_report.get('method', 'unknown')}",
        f"- status: {solver_report.get('status', 'unknown')}",
        f"- mode: {solver_report.get('mode', 'n/a')}",
        f"- index mode: {solver_report.get('index_mode', 'n/a')}",
        f"- reverse: {solver_report.get('reverse', 'n/a')}",
        f"- pairs: {solver_report.get('num_pairs', 'n/a')}",
        f"- dimension: {solver_report.get('dimension', 'n/a')}",
        f"- differences shape: {differences_shape}",
        f"- norm: {_format_float(solver_report.get('norm'))}",
        (
            "- constraint satisfaction: "
            f"{_format_float(solver_report.get('constraint_satisfaction'))}"
        ),
        f"- margin satisfaction: {_format_float(solver_report.get('margin_satisfaction'))}",
        f"- objective: {_format_float(solver_report.get('objective'))}",
    ]

    if metadata:
        lines.extend(
            [
                "",
                "## Pair Materialization",
                "",
                f"- activation key: {metadata.get('activation_key', 'n/a')}",
                f"- activation items: {metadata.get('num_activation_items', 'n/a')}",
                f"- records consumed: {metadata.get('records_consumed', 'n/a')}",
                f"- records or lines skipped: {metadata.get('records_or_lines_skipped', 'n/a')}",
            ]
        )

    lines.extend(["", *_novelty_lines(novelty_report), ""])
    lines.extend([*_evaluation_lines(evaluation_report), ""])
    lines.extend([*_baseline_lines(baselines_report), ""])
    lines.extend([*_policy_margin_lines(policy_margin_report), ""])
    lines.extend(
        [
            "## Pair Examples",
            "",
            "| # | best | subpar | delta cp | best PV | subpar PV | root FEN |",
            "|---|---|---|---:|---|---|---|",
        ]
    )

    if top_n < 0:
        raise ValueError("top_n must be >= 0")
    available = max(
        len(pairs["root_fens"]),
        len(pairs["best_moves"]),
        len(pairs["subpar_moves"]),
        len(pairs["best_pv"]),
        len(pairs["subpar_pv"]),
    )
    count = min(top_n, available)
    for idx in range(count):
        best_score = _list_get(pairs["best_score_cp"], idx, None)
        subpar_score = _list_get(pairs["subpar_score_cp"], idx, None)
        delta = "n/a"
        if best_score is not None and subpar_score is not None:
            try:
                delta = str(int(best_score) - int(subpar_score))
            except (TypeError, ValueError):
                delta = "n/a"
        lines.append(
            "| {idx} | {best} ({best_score}) | {subpar} ({subpar_score}) | "
            "{delta} | {best_pv} | {subpar_pv} | {root} |".format(
                idx=idx,
                best=_escape_table(_list_get(pairs["best_moves"], idx, "")),
                best_score=_escape_table(best_score),
                subpar=_escape_table(_list_get(pairs["subpar_moves"], idx, "")),
                subpar_score=_escape_table(subpar_score),
                delta=_escape_table(delta),
                best_pv=_escape_table(_list_get(pairs["best_pv"], idx, "")),
                subpar_pv=_escape_table(_list_get(pairs["subpar_pv"], idx, "")),
                root=_escape_table(_list_get(pairs["root_fens"], idx, "")),
            )
        )
    if count == 0:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    return "\n".join(lines).strip() + "\n"
