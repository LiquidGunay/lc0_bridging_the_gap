"""Novelty metrics for machine-vs-human activation subspaces."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def as_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return embeddings as a rank-2 matrix, flattening token activations when needed."""
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim == 3:
        return arr.reshape((arr.shape[0], -1))
    if arr.ndim != 2:
        raise ValueError(f"Expected rank-2 embeddings or rank-3 token activations, got {arr.shape}")
    return arr


def right_svd_basis(
    embeddings: np.ndarray,
    *,
    max_rank: int | None = None,
    center: bool = True,
) -> np.ndarray:
    """Compute right singular vectors as row vectors in feature space."""
    matrix = as_embedding_matrix(embeddings)
    if center:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    if max_rank is not None:
        max_rank = max(1, min(max_rank, min(matrix.shape)))
    _u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
    return vh[:max_rank] if max_rank is not None else vh


def reconstruction_loss(vector: np.ndarray, basis: np.ndarray, *, rank: int) -> float:
    """Return normalized squared reconstruction loss from a truncated SVD basis."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(f"Expected rank-2 basis, got {basis.shape}")
    if vec.shape[0] != basis.shape[1]:
        raise ValueError(f"Vector dimension {vec.shape[0]} does not match basis {basis.shape[1]}")
    denom = float(np.dot(vec, vec))
    if denom == 0.0:
        return float("nan")
    rank = max(1, min(rank, basis.shape[0]))
    truncated = basis[:rank]
    projection = truncated.T @ (truncated @ vec)
    residual = vec - projection
    return float(np.dot(residual, residual) / denom)


def novelty_curve(
    concept_vectors: np.ndarray,
    machine_embeddings: np.ndarray,
    human_embeddings: np.ndarray,
    *,
    ranks: Iterable[int] = (32, 64, 128, 256, 512, 1024),
    center: bool = True,
) -> list[dict]:
    """Compare concept reconstruction under machine and human activation bases."""
    machine = as_embedding_matrix(machine_embeddings)
    human = as_embedding_matrix(human_embeddings)
    directions = np.asarray(concept_vectors, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions[:, None]
    if directions.ndim != 2:
        raise ValueError(f"Expected concept vectors shaped [d] or [d, k], got {directions.shape}")
    if directions.shape[0] != machine.shape[1] or directions.shape[0] != human.shape[1]:
        raise ValueError(
            "Concept and embedding dimensions must match: "
            f"concept={directions.shape[0]}, machine={machine.shape[1]}, human={human.shape[1]}"
        )

    ranks = [int(rank) for rank in ranks]
    max_rank = max(ranks)
    machine_basis = right_svd_basis(machine, max_rank=max_rank, center=center)
    human_basis = right_svd_basis(human, max_rank=max_rank, center=center)

    reports = []
    for idx in range(directions.shape[1]):
        vec = directions[:, idx]
        curve = []
        novelty_scores = []
        for rank in ranks:
            machine_loss = reconstruction_loss(vec, machine_basis, rank=rank)
            human_loss = reconstruction_loss(vec, human_basis, rank=rank)
            novelty = human_loss - machine_loss
            novelty_scores.append(novelty)
            curve.append(
                {
                    "rank": rank,
                    "machine_loss": machine_loss,
                    "human_loss": human_loss,
                    "novelty": novelty,
                }
            )
        reports.append(
            {
                "vector": idx,
                "curve": curve,
                "novelty_area": float(np.nanmean(novelty_scores)),
                "positive_rank_fraction": float(np.mean(np.asarray(novelty_scores) > 0)),
            }
        )
    return reports
