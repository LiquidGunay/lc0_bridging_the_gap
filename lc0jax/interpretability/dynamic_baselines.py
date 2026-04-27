"""Baselines for dynamic sparse concept directions."""

from __future__ import annotations

from typing import Any

import numpy as np

from lc0jax.interpretability.concepts import solve_sparse_concept_from_differences


def evaluate_direction(
    differences: np.ndarray,
    direction: np.ndarray,
    *,
    margin: float = 1.0,
) -> dict[str, Any]:
    """Evaluate constraint satisfaction for one direction."""
    differences = np.asarray(differences, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    if differences.ndim != 2:
        raise ValueError(f"Expected rank-2 differences, got {differences.shape}")
    if differences.shape[0] == 0:
        raise ValueError("At least one difference row is required")
    if differences.shape[1] == 0:
        raise ValueError("At least one feature column is required")
    if direction.shape[0] != differences.shape[1]:
        raise ValueError(
            f"Direction dim {direction.shape[0]} does not match differences dim {differences.shape[1]}"
        )
    scores = differences @ direction
    return {
        "count": int(scores.shape[0]),
        "mean_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "constraint_satisfaction": float(np.mean(scores > 0)),
        "margin_satisfaction": float(np.mean(scores >= margin)),
        "norm": float(np.linalg.norm(direction)),
    }


def random_sparse_directions(
    *,
    count: int,
    dimension: int,
    nonzero: int,
    seed: int,
) -> np.ndarray:
    """Generate unit-norm random sparse directions."""
    if count < 0:
        raise ValueError("count must be >= 0")
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    nonzero = max(1, min(int(nonzero), int(dimension)))
    rng = np.random.default_rng(seed)
    directions = np.zeros((count, dimension), dtype=np.float64)
    for idx in range(count):
        support = rng.choice(dimension, size=nonzero, replace=False)
        values = rng.standard_normal(nonzero)
        norm = np.linalg.norm(values)
        if norm == 0:
            values[0] = 1.0
            norm = 1.0
        directions[idx, support] = values / norm
    return directions


def _summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {"count": 0}
    keys = [
        "mean_score",
        "min_score",
        "max_score",
        "constraint_satisfaction",
        "margin_satisfaction",
        "norm",
    ]
    summary: dict[str, Any] = {"count": len(metrics)}
    for key in keys:
        values = np.asarray([item[key] for item in metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_max"] = float(np.max(values))
    return summary


def dynamic_baseline_report(
    differences: np.ndarray,
    direction: np.ndarray,
    *,
    margin: float = 1.0,
    random_count: int = 128,
    shuffled_label_count: int = 128,
    shuffled_solve_count: int = 0,
    nonzero_threshold: float = 1e-8,
    seed: int = 0,
    c: float = 1.0,
    standardize: bool = True,
) -> dict[str, Any]:
    """Build random and shuffled-label baselines for a dynamic concept."""
    differences = np.asarray(differences, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    actual = evaluate_direction(differences, direction, margin=margin)
    nonzero = int(np.count_nonzero(np.abs(direction) > nonzero_threshold))
    nonzero = max(1, nonzero)

    random_dirs = random_sparse_directions(
        count=random_count,
        dimension=differences.shape[1],
        nonzero=nonzero,
        seed=seed,
    )
    random_metrics = [
        evaluate_direction(differences, random_dir, margin=margin)
        for random_dir in random_dirs
    ]

    rng = np.random.default_rng(seed + 1)
    shuffled_label_metrics = []
    for _ in range(shuffled_label_count):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=differences.shape[0])
        shuffled_label_metrics.append(
            evaluate_direction(differences * signs[:, None], direction, margin=margin)
        )

    shuffled_solve_metrics = []
    for _ in range(shuffled_solve_count):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=differences.shape[0])
        solved = solve_sparse_concept_from_differences(
            differences * signs[:, None],
            c=c,
            margin=margin,
            standardize=standardize,
        )
        shuffled_solve_metrics.append(
            evaluate_direction(differences, solved["direction"], margin=margin)
        )

    return {
        "num_pairs": int(differences.shape[0]),
        "dimension": int(differences.shape[1]),
        "margin": float(margin),
        "seed": int(seed),
        "nonzero_features": int(nonzero),
        "actual": actual,
        "random_sparse": _summarize_metrics(random_metrics),
        "shuffled_labels": _summarize_metrics(shuffled_label_metrics),
        "shuffled_solve": _summarize_metrics(shuffled_solve_metrics),
    }
