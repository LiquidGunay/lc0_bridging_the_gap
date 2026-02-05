"""Elo math helpers for lightweight benchmarks."""

from __future__ import annotations

import math


def elo_from_score(score: float, *, eps: float = 1e-6) -> float:
    """Convert a match score in (0,1) to an Elo difference."""
    score = min(max(score, eps), 1.0 - eps)
    return 400.0 * math.log10(score / (1.0 - score))


def score_ci(score: float, games: int) -> tuple[float, float]:
    """Normal-approx score CI for quick diagnostics."""
    if games <= 0:
        return 0.0, 1.0
    var = score * (1.0 - score) / games
    stderr = math.sqrt(max(var, 0.0))
    low = max(score - 1.96 * stderr, 0.0)
    high = min(score + 1.96 * stderr, 1.0)
    return low, high
