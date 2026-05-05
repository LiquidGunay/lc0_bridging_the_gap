"""Concept-family discovery over dynamic rollout differences."""

from __future__ import annotations

from typing import Any

import numpy as np

from .concepts import (
    solve_screened_sparse_concept_from_differences,
    solve_sparse_concept_from_differences,
)


def _normalize_rows(rows: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms < epsilon, 1.0, norms)


def _cosine(a: np.ndarray, b: np.ndarray, *, epsilon: float = 1e-12) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < epsilon:
        return 0.0
    return float(np.dot(a, b) / denom)


def cluster_difference_rows(
    differences: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 0,
    normalize_rows: bool = True,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster rollout-difference rows and return labels plus centers."""
    from sklearn.cluster import MiniBatchKMeans

    diff = np.asarray(differences, dtype=np.float64)
    if diff.ndim != 2:
        raise ValueError(f"Expected a rank-2 difference matrix, got {diff.shape}")
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if diff.shape[0] < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of difference rows")

    clustering_input = _normalize_rows(diff) if normalize_rows else diff
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=5,
        random_state=seed,
    )
    labels = km.fit_predict(clustering_input)
    return labels.astype(np.int32), np.asarray(km.cluster_centers_, dtype=np.float64)


def _solve_family_direction(
    differences: np.ndarray,
    *,
    max_features: int | None,
    screening_method: str,
    c: float,
    margin: float,
    standardize: bool,
) -> dict[str, Any]:
    if max_features is None or max_features >= differences.shape[1]:
        return solve_sparse_concept_from_differences(
            differences,
            c=c,
            margin=margin,
            standardize=standardize,
        )
    return solve_screened_sparse_concept_from_differences(
        differences,
        max_features=max_features,
        screening_method=screening_method,
        c=c,
        margin=margin,
        standardize=standardize,
    )


def bootstrap_family_stability(
    differences: np.ndarray,
    reference_direction: np.ndarray,
    *,
    max_features: int | None,
    screening_method: str,
    c: float,
    margin: float,
    standardize: bool,
    bootstrap_count: int,
    bootstrap_fraction: float,
    min_rows: int,
    seed: int,
    cosine_threshold: float,
) -> dict[str, Any]:
    """Solve bootstrap resamples and compare their directions to a reference."""
    diff = np.asarray(differences, dtype=np.float64)
    if bootstrap_count <= 0:
        return {"enabled": False, "requested": int(bootstrap_count)}
    if diff.shape[0] < min_rows:
        return {
            "enabled": False,
            "requested": int(bootstrap_count),
            "reason": "too_few_rows",
            "num_rows": int(diff.shape[0]),
            "min_rows": int(min_rows),
        }

    rng = np.random.default_rng(seed)
    sample_size = int(round(diff.shape[0] * bootstrap_fraction))
    sample_size = min(diff.shape[0], max(min_rows, sample_size))
    cosines = []
    failures = 0
    for _ in range(bootstrap_count):
        sample = rng.choice(diff.shape[0], size=sample_size, replace=True)
        try:
            result = _solve_family_direction(
                diff[sample],
                max_features=max_features,
                screening_method=screening_method,
                c=c,
                margin=margin,
                standardize=standardize,
            )
        except Exception:
            failures += 1
            continue
        cosines.append(_cosine(reference_direction, result["direction"]))

    cosine_array = np.asarray(cosines, dtype=np.float64)
    if cosine_array.size == 0:
        return {
            "enabled": True,
            "requested": int(bootstrap_count),
            "completed": 0,
            "failed": int(failures),
        }
    return {
        "enabled": True,
        "requested": int(bootstrap_count),
        "completed": int(cosine_array.size),
        "failed": int(failures),
        "sample_size": int(sample_size),
        "mean_cosine": float(np.mean(cosine_array)),
        "min_cosine": float(np.min(cosine_array)),
        "std_cosine": float(np.std(cosine_array)),
        "pass_fraction": float(np.mean(cosine_array >= cosine_threshold)),
        "cosine_threshold": float(cosine_threshold),
    }


def solve_dynamic_concept_families(
    differences: np.ndarray,
    *,
    n_clusters: int,
    min_cluster_size: int = 4,
    max_features: int | None = 2048,
    screening_method: str = "abs_mean",
    c: float = 1.0,
    margin: float = 1.0,
    standardize: bool = True,
    seed: int = 0,
    normalize_rows: bool = True,
    bootstrap_count: int = 0,
    bootstrap_fraction: float = 0.8,
    bootstrap_min_rows: int = 4,
    bootstrap_cosine_threshold: float = 0.7,
) -> dict[str, Any]:
    """Cluster dynamic differences and solve one sparse concept per cluster."""
    diff = np.asarray(differences, dtype=np.float64)
    labels, centers = cluster_difference_rows(
        diff,
        n_clusters=n_clusters,
        seed=seed,
        normalize_rows=normalize_rows,
    )

    families = []
    skipped = []
    family_ids = []
    cluster_ids = []
    directions = []
    raw_directions = []
    row_indices = []
    for cluster_id in range(n_clusters):
        indices = np.flatnonzero(labels == cluster_id)
        if indices.shape[0] < min_cluster_size:
            skipped.append(
                {
                    "cluster_id": int(cluster_id),
                    "num_rows": int(indices.shape[0]),
                    "reason": "below_min_cluster_size",
                }
            )
            continue

        result = _solve_family_direction(
            diff[indices],
            max_features=max_features,
            screening_method=screening_method,
            c=c,
            margin=margin,
            standardize=standardize,
        )
        family_id = len(families)
        stability = bootstrap_family_stability(
            diff[indices],
            result["direction"],
            max_features=max_features,
            screening_method=screening_method,
            c=c,
            margin=margin,
            standardize=standardize,
            bootstrap_count=bootstrap_count,
            bootstrap_fraction=bootstrap_fraction,
            min_rows=bootstrap_min_rows,
            seed=seed + 1009 + cluster_id,
            cosine_threshold=bootstrap_cosine_threshold,
        )
        directions.append(result["direction"])
        raw_directions.append(result["raw_direction"])
        family_ids.append(family_id)
        cluster_ids.append(cluster_id)
        row_indices.append(indices.astype(np.int64))
        families.append(
            {
                "family_id": int(family_id),
                "cluster_id": int(cluster_id),
                "num_rows": int(indices.shape[0]),
                "row_indices": indices.astype(np.int64),
                "centroid_norm": float(np.linalg.norm(centers[cluster_id])),
                "method": result["method"],
                "status": result["status"],
                "norm": result["norm"],
                "constraint_satisfaction": result["constraint_satisfaction"],
                "margin_satisfaction": result["margin_satisfaction"],
                "objective": result["objective"],
                "screening_enabled": bool(result.get("screening_enabled", False)),
                "screening_method": result.get("screening_method"),
                "screening_max_features": result.get("screening_max_features"),
                "screened_dimension": result.get("screened_dimension", diff.shape[1]),
                "screening_indices": result.get("screening_indices"),
                "screening_scores": result.get("screening_scores"),
                "stability": stability,
            }
        )

    if not families:
        raise ValueError("No clusters met min_cluster_size")

    return {
        "labels": labels,
        "centers": centers,
        "families": families,
        "skipped_clusters": skipped,
        "family_ids": np.asarray(family_ids, dtype=np.int32),
        "cluster_ids": np.asarray(cluster_ids, dtype=np.int32),
        "directions": np.asarray(directions, dtype=np.float64),
        "raw_directions": np.asarray(raw_directions, dtype=np.float64),
        "row_indices": np.asarray(row_indices, dtype=object),
    }
