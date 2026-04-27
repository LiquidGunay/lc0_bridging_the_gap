"""Build sparse-concept constraint matrices from rollout metadata."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

import numpy as np

from lc0jax.interpretability.concepts import aggregate_trajectory


def iter_rollout_pair_records(path: str | Path) -> Iterable[dict]:
    """Yield rollout-pair records from the JSONL written by ``build_mcts_pairs.py``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_activation_index(
    path: str | Path,
    *,
    activation_key: str = "auto",
) -> tuple[dict[str, np.ndarray], str]:
    """Load activation shards into a FEN-to-activation index.

    ``activation_key="auto"`` prefers raw ``token_activations`` when present and
    falls back to ``embeddings`` for older shards.
    """
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No activation shards found at {root}")

    index: dict[str, np.ndarray] = {}
    chosen_key: str | None = None
    for file in files:
        data = np.load(file, allow_pickle=True)
        if "fens" not in data:
            raise KeyError(f"Activation shard lacks fens: {file}")
        if activation_key == "auto":
            key = "token_activations" if "token_activations" in data else "embeddings"
        else:
            key = activation_key
        if key not in data:
            raise KeyError(f"Activation shard {file} lacks key '{key}'")
        if chosen_key is None:
            chosen_key = key
        elif chosen_key != key:
            raise ValueError(
                f"Activation key changed across shards: {chosen_key} then {key}"
            )
        activations = data[key]
        fens = data["fens"].tolist()
        if len(fens) != activations.shape[0]:
            raise ValueError(
                f"Shard {file} has {len(fens)} FENs but {activations.shape[0]} activations"
            )
        for fen, activation in zip(fens, activations):
            index.setdefault(str(fen), np.asarray(activation))

    if chosen_key is None:
        raise RuntimeError(f"No activations loaded from {root}")
    return index, chosen_key


def trajectory_activations(
    fens: list[str],
    activation_index: dict[str, np.ndarray],
) -> np.ndarray:
    """Return activations for all FENs in one rollout trajectory."""
    missing = [fen for fen in fens if fen not in activation_index]
    if missing:
        raise KeyError(f"Missing activations for {len(missing)} trajectory FENs")
    return np.asarray([activation_index[fen] for fen in fens])


def materialize_rollout_differences(
    records: Iterable[dict],
    activation_index: dict[str, np.ndarray],
    *,
    mode: str = "flat",
    index_mode: str = "both",
    max_records: int | None = None,
) -> dict:
    """Build ``psi(best) - psi(subpar)`` rows and aligned metadata arrays."""
    differences = []
    root_fens = []
    best_moves = []
    subpar_moves = []
    best_scores = []
    subpar_scores = []
    best_pvs = []
    subpar_pvs = []
    skipped = 0
    consumed = 0

    for record in records:
        if max_records is not None and consumed >= max_records:
            break
        consumed += 1
        try:
            best_line = record["best"]
            best_activation = trajectory_activations(best_line["fens"], activation_index)
            best_vector = aggregate_trajectory(
                best_activation,
                mode=mode,
                index_mode=index_mode,
            )
        except (KeyError, ValueError):
            skipped += 1
            continue

        for subpar_line in record.get("subpar", []):
            try:
                subpar_activation = trajectory_activations(
                    subpar_line["fens"],
                    activation_index,
                )
                subpar_vector = aggregate_trajectory(
                    subpar_activation,
                    mode=mode,
                    index_mode=index_mode,
                )
            except (KeyError, ValueError):
                skipped += 1
                continue

            differences.append(best_vector - subpar_vector)
            root_fens.append(record.get("root_fen", ""))
            best_moves.append(best_line.get("move", ""))
            subpar_moves.append(subpar_line.get("move", ""))
            best_scores.append(best_line.get("score_cp"))
            subpar_scores.append(subpar_line.get("score_cp"))
            best_pvs.append(" ".join(best_line.get("pv", [])))
            subpar_pvs.append(" ".join(subpar_line.get("pv", [])))

    if not differences:
        raise ValueError("No rollout differences could be materialized")

    return {
        "differences": np.asarray(differences),
        "root_fens": np.asarray(root_fens, dtype=object),
        "best_moves": np.asarray(best_moves, dtype=object),
        "subpar_moves": np.asarray(subpar_moves, dtype=object),
        "best_score_cp": np.asarray(best_scores, dtype=object),
        "subpar_score_cp": np.asarray(subpar_scores, dtype=object),
        "best_pv": np.asarray(best_pvs, dtype=object),
        "subpar_pv": np.asarray(subpar_pvs, dtype=object),
        "records_consumed": np.asarray(consumed, dtype=np.int32),
        "records_or_lines_skipped": np.asarray(skipped, dtype=np.int32),
    }
