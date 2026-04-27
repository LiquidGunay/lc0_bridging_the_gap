"""Shared loaders for dynamic concept artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_concept_direction(path: str | Path) -> np.ndarray:
    """Load a single dynamic concept direction from a concept dir or NPZ file."""
    path = Path(path)
    concept_file = path / "concept_direction.npz" if path.is_dir() else path
    data = np.load(concept_file, allow_pickle=True)
    direction = data["direction"]
    if direction.ndim == 2:
        if direction.shape[1] != 1:
            raise ValueError("Expected exactly one concept direction")
        direction = direction[:, 0]
    return np.asarray(direction)


def load_pair_differences(path: str | Path) -> np.ndarray:
    """Load the materialized pair-difference matrix from a pairs NPZ file."""
    data = np.load(path, allow_pickle=True)
    if "differences" not in data:
        raise KeyError("pairs.npz must contain materialized 'differences'")
    return np.asarray(data["differences"])
