"""Concept discovery and activation patching."""

from __future__ import annotations

import numpy as np

from lc0jax.modeling.encode import encode_board
from lc0jax.modeling.inference import forward

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None

def discover_concepts(
    embeddings_a,
    embeddings_b,
    *,
    method: str = "mean_diff",
    shrinkage: float = 1e-3,
    k: int = 8,
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
        import cvxpy as cp

        # Subsample to keep cvxpy fast, random pairing
        n_samples = min(len(emb_a), len(emb_b), 2000)
        rng = np.random.default_rng(42)
        idx_a = rng.choice(len(emb_a), n_samples, replace=False)
        idx_b = rng.choice(len(emb_b), n_samples, replace=False)

        X_pos = emb_a[idx_a]
        X_neg = emb_b[idx_b]

        d = X_pos.shape[1]
        v = cp.Variable(d)
        xi = cp.Variable(n_samples)

        # L1 penalized SVM with soft-margin to avoid infeasibility
        C = 1.0
        objective = cp.Minimize(cp.norm1(v) + C * cp.sum(xi))
        constraints = [
            (X_pos - X_neg) @ v >= 1 - xi,
            xi >= 0
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        if v.value is None:
            raise RuntimeError("cvxpy could not find a solution for svm_cvxpy")

        vec = v.value
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
    labels = np.concatenate([np.zeros(len(emb_a), dtype=np.int32), np.ones(len(emb_b), dtype=np.int32)])

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
