"""Prototype selection for dynamic concept teachability workflows."""

from __future__ import annotations

from typing import Any

import numpy as np


PAIR_METADATA_KEYS = (
    "root_fens",
    "best_moves",
    "subpar_moves",
    "best_score_cp",
    "subpar_score_cp",
    "best_pv",
    "subpar_pv",
)


def projection_scores(differences: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project dynamic pair differences onto a concept direction."""
    differences = np.asarray(differences, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    if differences.ndim != 2:
        raise ValueError(f"Expected rank-2 differences, got {differences.shape}")
    if direction.shape[0] != differences.shape[1]:
        raise ValueError(
            f"Direction dim {direction.shape[0]} does not match "
            f"differences dim {differences.shape[1]}"
        )
    return differences @ direction


def select_top_indices(
    scores: np.ndarray,
    *,
    top_k: int,
    largest: bool = True,
) -> np.ndarray:
    """Return deterministic score-sorted prototype indices."""
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    count = min(int(top_k), int(scores.shape[0]))
    order = np.argsort(-scores if largest else scores, kind="stable")
    return order[:count].astype(np.int64)


def select_random_indices(
    *,
    num_rows: int,
    count: int,
    seed: int,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Sample random control indices without replacement."""
    if num_rows < 0:
        raise ValueError("num_rows must be >= 0")
    if count < 0:
        raise ValueError("count must be >= 0")
    excluded = set() if exclude is None else set(np.asarray(exclude, dtype=np.int64).tolist())
    pool = np.asarray([idx for idx in range(num_rows) if idx not in excluded], dtype=np.int64)
    if pool.size == 0 or count == 0:
        return np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=min(count, int(pool.size)), replace=False)
    return np.asarray(chosen, dtype=np.int64)


def _as_list(pair_metadata: dict[str, Any], key: str) -> list[Any]:
    if key not in pair_metadata:
        return []
    value = np.asarray(pair_metadata[key], dtype=object)
    if value.ndim == 0:
        return []
    return value.tolist()


def _list_get(items: list[Any], idx: int, default: Any = "") -> Any:
    return items[idx] if idx < len(items) else default


def _prototype_rows(
    pair_metadata: dict[str, Any],
    indices: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    metadata_lists = {key: _as_list(pair_metadata, key) for key in PAIR_METADATA_KEYS}
    rows = []
    for rank, idx in enumerate(np.asarray(indices, dtype=np.int64).tolist()):
        row = {
            "rank": int(rank),
            "index": int(idx),
            "score": float(scores[idx]),
        }
        for key, values in metadata_lists.items():
            row[key] = _list_get(values, idx, "")
        rows.append(row)
    return rows


def dynamic_prototype_report(
    differences: np.ndarray,
    direction: np.ndarray,
    pair_metadata: dict[str, Any],
    *,
    top_k: int = 32,
    random_count: int = 32,
    seed: int = 0,
    split_name: str = "train",
    direction_key: str = "direction",
) -> dict[str, Any]:
    """Build a prototype and random-control report for a dynamic concept."""
    scores = projection_scores(differences, direction)
    if scores.size == 0:
        raise ValueError("At least one pair row is required for prototype selection")
    top_indices = select_top_indices(scores, top_k=top_k, largest=True)
    random_indices = select_random_indices(
        num_rows=int(scores.shape[0]),
        count=random_count,
        seed=seed,
        exclude=top_indices,
    )
    return {
        "method": "dynamic_prototype_selection",
        "split": split_name,
        "direction_key": direction_key,
        "num_pairs": int(scores.shape[0]),
        "dimension": int(np.asarray(differences).shape[1]),
        "top_k": int(top_indices.shape[0]),
        "random_count": int(random_indices.shape[0]),
        "seed": int(seed),
        "score_summary": {
            "mean": float(np.mean(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
        "prototypes": _prototype_rows(pair_metadata, top_indices, scores),
        "random_controls": _prototype_rows(pair_metadata, random_indices, scores),
    }
