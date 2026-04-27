"""Held-out splits for dynamic rollout-pair datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def split_pair_indices(
    root_fens: list[str] | np.ndarray,
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test row indices grouped by root FEN.

    All rows with the same root FEN are assigned to the same split so multiple
    subpar alternatives from one search root cannot leak across train/test.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    roots = [str(item) for item in np.asarray(root_fens, dtype=object).reshape(-1)]
    if not roots:
        raise ValueError("At least one root FEN is required")

    unique_roots = list(dict.fromkeys(roots))
    if len(unique_roots) < 2:
        raise ValueError("At least two unique root FENs are required for a held-out split")

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(len(unique_roots))
    test_root_count = int(round(len(unique_roots) * test_fraction))
    test_root_count = min(max(1, test_root_count), len(unique_roots) - 1)
    test_roots = {unique_roots[int(idx)] for idx in permuted[:test_root_count]}

    train_indices = [idx for idx, root in enumerate(roots) if root not in test_roots]
    test_indices = [idx for idx, root in enumerate(roots) if root in test_roots]
    if not train_indices or not test_indices:
        raise RuntimeError("Split produced an empty train or test set")
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


def infer_pair_row_count(
    payload: Mapping[str, Any],
    *,
    row_key: str = "differences",
) -> int:
    """Infer row count from the main pair matrix."""
    if row_key not in payload:
        raise KeyError(f"pairs payload missing required row key '{row_key}'")
    rows = np.asarray(payload[row_key])
    if rows.ndim == 0:
        raise ValueError(f"row key '{row_key}' must have at least one dimension")
    return int(rows.shape[0])


def subset_pairs_payload(
    payload: Mapping[str, Any],
    indices: list[int] | np.ndarray,
    *,
    row_count: int | None = None,
    row_key: str = "differences",
) -> dict[str, np.ndarray]:
    """Subset row-aligned arrays while preserving scalar and non-row metadata."""
    if row_count is None:
        row_count = infer_pair_row_count(payload, row_key=row_key)
    if row_count < 0:
        raise ValueError("row_count must be non-negative")

    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if selected.size and (np.min(selected) < 0 or np.max(selected) >= row_count):
        raise IndexError("subset index out of bounds")

    subset: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == row_count:
            subset[key] = array[selected]
        else:
            subset[key] = array
    return subset


def root_split_summary(
    root_fens: list[str] | np.ndarray,
    train_indices: list[int] | np.ndarray,
    test_indices: list[int] | np.ndarray,
) -> dict[str, int]:
    """Summarize row and root counts for a grouped split."""
    roots = np.asarray(root_fens, dtype=object).reshape(-1)
    train = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    test = np.asarray(test_indices, dtype=np.int64).reshape(-1)
    return {
        "num_rows": int(roots.shape[0]),
        "num_train_rows": int(train.shape[0]),
        "num_test_rows": int(test.shape[0]),
        "num_root_groups": int(len(set(str(root) for root in roots.tolist()))),
        "num_train_root_groups": int(len(set(str(roots[idx]) for idx in train.tolist()))),
        "num_test_root_groups": int(len(set(str(roots[idx]) for idx in test.tolist()))),
    }
