"""Evaluation reports for dynamic concept directions."""

from __future__ import annotations

from typing import Any

import numpy as np

from lc0jax.interpretability.dynamic_baselines import evaluate_direction


def dynamic_evaluation_report(
    differences: np.ndarray,
    direction: np.ndarray,
    *,
    margin: float = 1.0,
    split_name: str = "heldout",
    nonzero_threshold: float = 1e-8,
) -> dict[str, Any]:
    """Evaluate a learned dynamic direction on one pair split."""
    differences = np.asarray(differences, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    metrics = evaluate_direction(differences, direction, margin=margin)
    nonzero = int(np.count_nonzero(np.abs(direction) > nonzero_threshold))
    return {
        "method": "dynamic_direction_evaluation",
        "split": split_name,
        "num_pairs": int(differences.shape[0]),
        "dimension": int(differences.shape[1]),
        "margin": float(margin),
        "nonzero_features": int(nonzero),
        "evaluation": metrics,
    }
