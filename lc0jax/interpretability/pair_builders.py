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
    """Load activation shards into a trajectory-key or FEN-to-activation index.

    ``activation_key="auto"`` prefers raw ``token_activations`` when present and
    falls back to ``embeddings`` for older shards. Shards with non-empty
    ``activation_keys`` are indexed by those stable keys; older shards are
    indexed by FEN for backwards compatibility.
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
        fens = [str(fen) for fen in data["fens"].tolist()]
        if len(fens) != activations.shape[0]:
            raise ValueError(
                f"Shard {file} has {len(fens)} FENs but {activations.shape[0]} activations"
            )
        activation_keys = []
        if "activation_keys" in data:
            activation_keys = [str(item) for item in data["activation_keys"].tolist()]
            if len(activation_keys) != activations.shape[0]:
                raise ValueError(
                    f"Shard {file} has {len(activation_keys)} activation keys "
                    f"but {activations.shape[0]} activations"
                )
        use_activation_keys = bool(activation_keys) and any(activation_keys)
        index_keys = activation_keys if use_activation_keys else fens
        for index_key, activation in zip(index_keys, activations):
            if not index_key:
                continue
            index.setdefault(index_key, np.asarray(activation))

    if chosen_key is None:
        raise RuntimeError(f"No activations loaded from {root}")
    return index, chosen_key


def trajectory_activations(
    trajectory_keys: list[str],
    activation_index: dict[str, np.ndarray],
) -> np.ndarray:
    """Return activations for all keys in one rollout trajectory."""
    missing = [key for key in trajectory_keys if key not in activation_index]
    if missing:
        raise KeyError(f"Missing activations for {len(missing)} trajectory positions")
    return np.asarray([activation_index[key] for key in trajectory_keys])


def trajectory_keys_for_line(line: dict) -> list[str]:
    """Return activation keys for a rollout line, falling back to FENs."""
    keys = [str(key) for key in line.get("activation_keys", []) if str(key)]
    if keys:
        return keys
    return [str(fen) for fen in line["fens"]]


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
            best_activation = trajectory_activations(
                trajectory_keys_for_line(best_line),
                activation_index,
            )
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
                    trajectory_keys_for_line(subpar_line),
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
