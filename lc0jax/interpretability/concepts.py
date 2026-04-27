"""Concept discovery and activation patching."""

from __future__ import annotations

import numpy as np

from lc0jax.modeling.encode import encode_board
from lc0jax.modeling.inference import forward

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None


def _normalize_vector(vec: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec, norm
    return vec / norm, norm


def _safe_feature_std(diff: np.ndarray, *, epsilon: float) -> np.ndarray:
    feature_std = diff.std(axis=0)
    return np.where(feature_std < epsilon, 1.0, feature_std)


def solve_sparse_concept_from_differences(
    differences: np.ndarray,
    *,
    c: float = 1.0,
    margin: float = 1.0,
    standardize: bool = True,
    epsilon: float = 1e-6,
    solver: str | None = None,
):
    """Solve the Schut-style sparse concept objective over paired differences.

    ``differences`` must contain rows ``phi(s+) - phi(s-)`` or
    ``psi(tau+) - psi(tau-)``. The optimization is:

    ``min ||v||_1 + c * sum(xi)`` subject to ``differences @ v >= margin - xi``.
    """
    import cvxpy as cp

    diff = np.asarray(differences, dtype=np.float64)
    if diff.ndim != 2:
        raise ValueError(f"Expected a rank-2 difference matrix, got {diff.shape}")
    if diff.shape[0] == 0:
        raise ValueError("At least one paired difference is required")

    if standardize:
        feature_std = _safe_feature_std(diff, epsilon=epsilon)
        solver_diff = diff / feature_std
    else:
        feature_std = np.ones(diff.shape[1], dtype=np.float64)
        solver_diff = diff

    v = cp.Variable(diff.shape[1])
    xi = cp.Variable(diff.shape[0])
    objective = cp.Minimize(cp.norm1(v) + c * cp.sum(xi))
    constraints = [
        solver_diff @ v >= margin - xi,
        xi >= 0,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver or cp.SCS, verbose=False)

    if v.value is None:
        raise RuntimeError("cvxpy could not find a sparse concept solution")

    standardized_raw = np.asarray(v.value, dtype=np.float64)
    raw = standardized_raw / feature_std
    direction, norm = _normalize_vector(raw)
    standardized_direction, standardized_norm = _normalize_vector(standardized_raw)
    margins = diff @ raw
    return {
        "direction": direction,
        "raw_direction": raw,
        "standardized_direction": standardized_direction,
        "standardized_raw_direction": standardized_raw,
        "feature_std": feature_std,
        "norm": norm,
        "standardized_norm": standardized_norm,
        "constraint_satisfaction": float(np.mean(margins > 0)),
        "margin_satisfaction": float(np.mean(margins >= margin - epsilon)),
        "margins": margins,
        "objective": float(prob.value) if prob.value is not None else None,
        "status": prob.status,
        "method": "sparse_cvxpy",
    }


def screen_sparse_concept_features(
    differences: np.ndarray,
    *,
    max_features: int,
    method: str = "abs_mean",
    standardize: bool = True,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic high-signal feature subset for large sparse solves."""
    diff = np.asarray(differences, dtype=np.float64)
    if diff.ndim != 2:
        raise ValueError(f"Expected a rank-2 difference matrix, got {diff.shape}")
    if diff.shape[0] == 0:
        raise ValueError("At least one paired difference is required")
    if diff.shape[1] == 0:
        raise ValueError("At least one feature column is required")
    if max_features < 1:
        raise ValueError("max_features must be >= 1")

    if standardize:
        feature_std = _safe_feature_std(diff, epsilon=epsilon)
        scoring_diff = diff / feature_std
    else:
        scoring_diff = diff

    if method == "abs_mean":
        scores = np.abs(scoring_diff.mean(axis=0))
    elif method == "mean_abs":
        scores = np.mean(np.abs(scoring_diff), axis=0)
    else:
        raise ValueError(f"Unsupported screening method: {method}")
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    keep = min(int(max_features), int(diff.shape[1]))
    order = np.lexsort((np.arange(scores.shape[0]), -scores))
    return np.sort(order[:keep]), scores


def solve_screened_sparse_concept_from_differences(
    differences: np.ndarray,
    *,
    max_features: int,
    screening_method: str = "abs_mean",
    c: float = 1.0,
    margin: float = 1.0,
    standardize: bool = True,
    epsilon: float = 1e-6,
    solver: str | None = None,
):
    """Solve the sparse concept objective after deterministic feature screening.

    The CVXPY objective is unchanged, but it is solved only on the selected
    feature subset. Returned vectors are expanded back to the original feature
    dimension so downstream patching and evaluation code can stay unchanged.
    """
    diff = np.asarray(differences, dtype=np.float64)
    selected, scores = screen_sparse_concept_features(
        diff,
        max_features=max_features,
        method=screening_method,
        standardize=standardize,
        epsilon=epsilon,
    )
    if selected.shape[0] == diff.shape[1]:
        result = solve_sparse_concept_from_differences(
            diff,
            c=c,
            margin=margin,
            standardize=standardize,
            epsilon=epsilon,
            solver=solver,
        )
        result.update(
            {
                "screening_enabled": False,
                "screening_method": screening_method,
                "screening_max_features": int(max_features),
                "screening_indices": selected,
                "screening_scores": scores[selected],
                "source_dimension": int(diff.shape[1]),
                "screened_dimension": int(diff.shape[1]),
            }
        )
        return result

    sub_result = solve_sparse_concept_from_differences(
        diff[:, selected],
        c=c,
        margin=margin,
        standardize=standardize,
        epsilon=epsilon,
        solver=solver,
    )

    raw = np.zeros(diff.shape[1], dtype=np.float64)
    raw[selected] = sub_result["raw_direction"]
    standardized_raw = np.zeros(diff.shape[1], dtype=np.float64)
    standardized_raw[selected] = sub_result["standardized_raw_direction"]
    if standardize:
        feature_std = _safe_feature_std(diff, epsilon=epsilon)
    else:
        feature_std = np.ones(diff.shape[1], dtype=np.float64)

    direction, norm = _normalize_vector(raw)
    standardized_direction, standardized_norm = _normalize_vector(standardized_raw)
    margins = diff @ raw
    return {
        "direction": direction,
        "raw_direction": raw,
        "standardized_direction": standardized_direction,
        "standardized_raw_direction": standardized_raw,
        "feature_std": feature_std,
        "norm": norm,
        "standardized_norm": standardized_norm,
        "constraint_satisfaction": float(np.mean(margins > 0)),
        "margin_satisfaction": float(np.mean(margins >= margin - epsilon)),
        "margins": margins,
        "objective": sub_result["objective"],
        "status": sub_result["status"],
        "method": "screened_sparse_cvxpy",
        "screening_enabled": True,
        "screening_method": screening_method,
        "screening_max_features": int(max_features),
        "screening_indices": selected,
        "screening_scores": scores[selected],
        "source_dimension": int(diff.shape[1]),
        "screened_dimension": int(selected.shape[0]),
    }


def aggregate_trajectory(
    activations: np.ndarray,
    *,
    mode: str = "flat",
    index_mode: str = "both",
) -> np.ndarray:
    """Aggregate a rollout's activation sequence into one trajectory vector.

    ``activations`` can be ``[T, 64, d]`` token activations or already-projected
    ``[T, d]`` embeddings. ``index_mode`` follows the Schut paper's distinction
    between both-player and single-player rollout indexing.
    """
    arr = np.asarray(activations)
    if index_mode == "both":
        pass
    elif index_mode == "single_even":
        arr = arr[::2]
    elif index_mode == "single_odd":
        arr = arr[1::2]
    else:
        raise ValueError(f"Unsupported index_mode: {index_mode}")
    if arr.shape[0] == 0:
        raise ValueError("Trajectory aggregation selected zero positions")

    if arr.ndim == 3:
        if arr.shape[1] != 64:
            raise ValueError(f"Expected token activations shaped [T, 64, d], got {arr.shape}")
        if mode == "flat":
            per_position = arr.reshape((arr.shape[0], -1))
        elif mode == "mean":
            per_position = arr.mean(axis=1)
        else:
            raise ValueError(f"Unsupported trajectory activation mode: {mode}")
    elif arr.ndim == 2:
        per_position = arr
    else:
        raise ValueError(f"Expected rank-2 or rank-3 trajectory activations, got {arr.shape}")
    return per_position.mean(axis=0)


def dynamic_rollout_differences(
    optimal_rollouts: np.ndarray,
    subpar_rollouts: np.ndarray,
    *,
    mode: str = "flat",
    index_mode: str = "both",
) -> np.ndarray:
    """Build ``psi(tau+) - psi(tau-)`` rows from stored rollout activations."""
    optimal = np.asarray(optimal_rollouts)
    subpar = np.asarray(subpar_rollouts)
    if optimal.shape[0] != subpar.shape[0]:
        raise ValueError(
            f"Optimal/subpar rollout count mismatch: {optimal.shape[0]} vs {subpar.shape[0]}"
        )

    rows = []
    for idx in range(optimal.shape[0]):
        pos = aggregate_trajectory(optimal[idx], mode=mode, index_mode=index_mode)
        neg_rollouts = subpar[idx]
        if neg_rollouts.ndim == optimal[idx].ndim:
            neg_rollouts = neg_rollouts[None, ...]
        for neg in neg_rollouts:
            rows.append(pos - aggregate_trajectory(neg, mode=mode, index_mode=index_mode))
    return np.asarray(rows)


def discover_concepts(
    embeddings_a,
    embeddings_b,
    *,
    method: str = "mean_diff",
    shrinkage: float = 1e-3,
    k: int = 8,
    standardize: bool = True,
):
    """Return a concept direction and summary stats."""
    emb_a = np.asarray(embeddings_a)
    emb_b = np.asarray(embeddings_b)
    mu_a = emb_a.mean(axis=0)
    mu_b = emb_b.mean(axis=0)
    scores = None

    if method == "mean_diff":
        vec = mu_a - mu_b
    elif method == "whitened_mean_diff":
        cov_a = np.cov(emb_a, rowvar=False)
        cov_b = np.cov(emb_b, rowvar=False)
        cov = (cov_a + cov_b) * 0.5
        cov = cov + np.eye(cov.shape[0]) * shrinkage
        vec = np.linalg.solve(cov, mu_a - mu_b)
    elif method == "cov_shift":
        cov_a = np.cov(emb_a, rowvar=False)
        cov_b = np.cov(emb_b, rowvar=False)
        delta = cov_a - cov_b
        eigvals, eigvecs = np.linalg.eigh(delta)
        idx = np.argsort(np.abs(eigvals))[::-1]
        vecs = eigvecs[:, idx[:k]]
        scores = eigvals[idx[:k]]
        vec = vecs
    elif method == "cluster_diff":
        vecs, scores = _cluster_diff_directions(emb_a, emb_b, k=k)
        vec = vecs
    elif method == "svm_cvxpy":
        # Subsample to keep cvxpy fast, random pairing
        n_samples = min(len(emb_a), len(emb_b), 2000)
        rng = np.random.default_rng(42)
        idx_a = rng.choice(len(emb_a), n_samples, replace=False)
        idx_b = rng.choice(len(emb_b), n_samples, replace=False)

        X_pos = emb_a[idx_a]
        X_neg = emb_b[idx_b]
        sparse = solve_sparse_concept_from_differences(
            X_pos - X_neg,
            c=1.0,
            margin=1.0,
            standardize=standardize,
        )
        vec = sparse["raw_direction"]
        scores = {
            "constraint_satisfaction": sparse["constraint_satisfaction"],
            "margin_satisfaction": sparse["margin_satisfaction"],
            "objective": sparse["objective"],
            "status": sparse["status"],
        }
    else:
        raise ValueError(f"Unsupported method: {method}")

    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        direction = vec / norm if norm > 0 else vec
    else:
        norms = np.linalg.norm(vec, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        direction = vec / norms
        norm = np.linalg.norm(vec, axis=0)
    return {
        "direction": direction,
        "raw_direction": vec,
        "mean_a": mu_a,
        "mean_b": mu_b,
        "norm": norm,
        "method": method,
        "scores": scores,
    }


def patch_activations(params, sample, concept_vec, *, alpha: float, layer: str = "trunk"):
    """Return output deltas after activation patching."""
    if chess is None:
        raise ImportError("python-chess is required for activation patching.")
    board = chess.Board(sample) if isinstance(sample, str) else sample
    planes = encode_board(board, [], planes_layout="nchw", input_format="INPUT_CLASSICAL_112_PLANE")
    base_policy, base_wdl, base_mlh = forward(params, planes)
    patched_policy, patched_wdl, patched_mlh = forward(
        params, planes, patch={"layer": layer, "vector": concept_vec, "alpha": alpha}
    )
    return {
        "delta_policy": np.asarray(patched_policy) - np.asarray(base_policy),
        "delta_wdl": np.asarray(patched_wdl) - np.asarray(base_wdl),
        "delta_mlh": np.asarray(patched_mlh) - np.asarray(base_mlh),
    }


def _cluster_diff_directions(emb_a: np.ndarray, emb_b: np.ndarray, *, k: int = 8):
    """Return k cluster-difference directions based on k-means over combined data."""
    from sklearn.cluster import MiniBatchKMeans

    emb_a = np.asarray(emb_a)
    emb_b = np.asarray(emb_b)
    X = np.concatenate([emb_a, emb_b], axis=0)
    labels = np.concatenate(
        [np.zeros(len(emb_a), dtype=np.int32), np.ones(len(emb_b), dtype=np.int32)]
    )

    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=5, random_state=0)
    assignments = km.fit_predict(X)
    centers = km.cluster_centers_

    scores = []
    for idx in range(k):
        mask = assignments == idx
        if mask.sum() == 0:
            scores.append(0.0)
            continue
        frac_a = (labels[mask] == 0).mean()
        scores.append(abs(frac_a - 0.5))

    vecs = centers.T
    scores = np.asarray(scores)
    order = np.argsort(scores)[::-1]
    return vecs[:, order], scores[order]
