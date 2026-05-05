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


def normalize_history_fens(value, root_fen: str) -> list[str]:
    """Return a FEN history list ending at ``root_fen``."""
    if value is None:
        items = []
    elif isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, str):
        items = [value] if value else []
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = []
    history = [str(item) for item in items if str(item)]
    root = str(root_fen)
    if not history or history[-1] != root:
        history.append(root)
    return history


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
    score_delta_cps = []
    best_wdls = []
    subpar_wdls = []
    best_multipv_ranks = []
    subpar_multipv_ranks = []
    best_depths = []
    subpar_depths = []
    best_nodes = []
    subpar_nodes = []
    best_seldepths = []
    subpar_seldepths = []
    best_nps = []
    subpar_nps = []
    best_hashfulls = []
    subpar_hashfulls = []
    best_tbhits = []
    subpar_tbhits = []
    best_raw_info_keys = []
    subpar_raw_info_keys = []
    best_pvs = []
    subpar_pvs = []
    root_history_fens = []
    root_game_ids = []
    root_game_indices = []
    root_plies = []
    root_sources = []
    root_record_ids = []
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
            score_delta_cps.append(subpar_line.get("score_delta_cp"))
            best_wdls.append(best_line.get("wdl"))
            subpar_wdls.append(subpar_line.get("wdl"))
            best_multipv_ranks.append(best_line.get("multipv_rank"))
            subpar_multipv_ranks.append(subpar_line.get("multipv_rank"))
            best_depths.append(best_line.get("depth"))
            subpar_depths.append(subpar_line.get("depth"))
            best_nodes.append(best_line.get("nodes"))
            subpar_nodes.append(subpar_line.get("nodes"))
            best_seldepths.append(best_line.get("seldepth"))
            subpar_seldepths.append(subpar_line.get("seldepth"))
            best_nps.append(best_line.get("nps"))
            subpar_nps.append(subpar_line.get("nps"))
            best_hashfulls.append(best_line.get("hashfull"))
            subpar_hashfulls.append(subpar_line.get("hashfull"))
            best_tbhits.append(best_line.get("tbhits"))
            subpar_tbhits.append(subpar_line.get("tbhits"))
            best_raw_info_keys.append(best_line.get("raw_info_keys", []))
            subpar_raw_info_keys.append(subpar_line.get("raw_info_keys", []))
            best_pvs.append(" ".join(best_line.get("pv", [])))
            subpar_pvs.append(" ".join(subpar_line.get("pv", [])))
            root_history_fens.append(record.get("root_history_fens", []))
            root_game_ids.append(record.get("root_game_id", ""))
            root_game_indices.append(record.get("root_game_index"))
            root_plies.append(record.get("root_ply"))
            root_sources.append(record.get("root_source", ""))
            root_record_ids.append(record.get("root_record_id", ""))

    if not differences:
        raise ValueError("No rollout differences could be materialized")

    return {
        "differences": np.asarray(differences),
        "root_fens": np.asarray(root_fens, dtype=object),
        "best_moves": np.asarray(best_moves, dtype=object),
        "subpar_moves": np.asarray(subpar_moves, dtype=object),
        "best_score_cp": np.asarray(best_scores, dtype=object),
        "subpar_score_cp": np.asarray(subpar_scores, dtype=object),
        "score_delta_cp": np.asarray(score_delta_cps, dtype=object),
        "best_wdl": np.asarray(best_wdls, dtype=object),
        "subpar_wdl": np.asarray(subpar_wdls, dtype=object),
        "best_multipv_rank": np.asarray(best_multipv_ranks, dtype=object),
        "subpar_multipv_rank": np.asarray(subpar_multipv_ranks, dtype=object),
        "best_depth": np.asarray(best_depths, dtype=object),
        "subpar_depth": np.asarray(subpar_depths, dtype=object),
        "best_nodes": np.asarray(best_nodes, dtype=object),
        "subpar_nodes": np.asarray(subpar_nodes, dtype=object),
        "best_seldepth": np.asarray(best_seldepths, dtype=object),
        "subpar_seldepth": np.asarray(subpar_seldepths, dtype=object),
        "best_nps": np.asarray(best_nps, dtype=object),
        "subpar_nps": np.asarray(subpar_nps, dtype=object),
        "best_hashfull": np.asarray(best_hashfulls, dtype=object),
        "subpar_hashfull": np.asarray(subpar_hashfulls, dtype=object),
        "best_tbhits": np.asarray(best_tbhits, dtype=object),
        "subpar_tbhits": np.asarray(subpar_tbhits, dtype=object),
        "best_raw_info_keys": np.asarray(best_raw_info_keys, dtype=object),
        "subpar_raw_info_keys": np.asarray(subpar_raw_info_keys, dtype=object),
        "best_pv": np.asarray(best_pvs, dtype=object),
        "subpar_pv": np.asarray(subpar_pvs, dtype=object),
        "root_history_fens": np.asarray(root_history_fens, dtype=object),
        "root_game_ids": np.asarray(root_game_ids, dtype=object),
        "root_game_indices": np.asarray(root_game_indices, dtype=object),
        "root_plies": np.asarray(root_plies, dtype=object),
        "root_sources": np.asarray(root_sources, dtype=object),
        "root_record_ids": np.asarray(root_record_ids, dtype=object),
        "records_consumed": np.asarray(consumed, dtype=np.int32),
        "records_or_lines_skipped": np.asarray(skipped, dtype=np.int32),
    }
