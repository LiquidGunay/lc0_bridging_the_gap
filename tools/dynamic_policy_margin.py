"""Measure dynamic concept patch effects on best-vs-subpar policy margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lc0jax.interpretability.dynamic_causal import (
    control_calibration_report,
    policy_margin_report,
)
from lc0jax.interpretability.pair_builders import normalize_history_fens
from lc0jax.modeling.encode import encode_board
from lc0jax.modeling.inference import forward
from lc0jax.modeling.policy import attention_policy_map, legal_move_mask, move_to_policy_index
from lc0jax.modeling.weights import load_pb_gz, map_bt4_weights

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None


def _load_direction(path: Path, *, vector: int) -> np.ndarray:
    concept_file = path / "concept_direction.npz" if path.is_dir() else path
    data = np.load(concept_file, allow_pickle=True)
    direction = data["direction"]
    if direction.ndim == 1:
        if vector != 0:
            raise IndexError("Only vector 0 is available for a single direction")
        return np.asarray(direction)
    return np.asarray(direction[:, vector])


def _load_pair_rows(path: Path, *, max_pairs: int | None, seed: int) -> dict:
    data = np.load(path, allow_pickle=True)
    required = ["root_fens", "best_moves", "subpar_moves"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"pairs.npz missing required metadata keys: {missing}")
    rows = {
        "root_fens": data["root_fens"].tolist(),
        "best_moves": data["best_moves"].tolist(),
        "subpar_moves": data["subpar_moves"].tolist(),
    }
    if "root_history_fens" in data:
        rows["root_history_fens"] = data["root_history_fens"].tolist()
    else:
        rows["root_history_fens"] = [[fen] for fen in rows["root_fens"]]
    count = min(len(rows["root_fens"]), len(rows["best_moves"]), len(rows["subpar_moves"]))
    indices = np.arange(count)
    if max_pairs is not None and count > max_pairs:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_pairs, replace=False))
    return {key: [values[idx] for idx in indices] for key, values in rows.items()}


def _batch_indices(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def _control_directions(
    direction: np.ndarray,
    *,
    kind: str,
    count: int,
    seed: int,
) -> list[tuple[str, int, np.ndarray]]:
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    base = np.asarray(direction, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(base))
    controls = []
    for idx in range(count):
        if kind == "random":
            control = rng.standard_normal(base.shape)
            control_norm = float(np.linalg.norm(control))
            if control_norm > 0.0:
                control = control * (norm / control_norm)
        elif kind == "shuffled":
            control = rng.permutation(base)
        else:
            raise ValueError(f"Unsupported control kind: {kind}")
        controls.append((kind, idx, control.reshape(direction.shape)))
    return controls


def _encode_plane_batches(valid_rows: dict, *, batch_size: int) -> list[np.ndarray]:
    batches = []
    for start, stop in _batch_indices(len(valid_rows["root_fens"]), batch_size):
        planes = []
        for fen, history_fens in zip(
            valid_rows["root_fens"][start:stop],
            valid_rows["root_history_fens"][start:stop],
        ):
            board = chess.Board(fen)
            history_boards = [chess.Board(history_fen) for history_fen in history_fens]
            planes.append(
                encode_board(
                    board,
                    history_boards,
                    planes_layout="nchw",
                    input_format="INPUT_CLASSICAL_112_PLANE",
                )
            )
        batches.append(np.stack(planes, axis=0))
    return batches


def _forward_policy_batches(
    params,
    plane_batches: list[np.ndarray],
    *,
    layer: str,
    direction: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    policies = []
    patch = None if direction is None else {"layer": layer, "vector": direction, "alpha": alpha}
    for planes in plane_batches:
        policy, _, _ = forward(params, planes, patch=patch)
        policies.append(np.asarray(policy))
    return np.concatenate(policies, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Solver-ready pairs.npz file.")
    parser.add_argument("--concept", required=True, help="Concept directory or direction npz.")
    parser.add_argument("--pb", required=True, help="Path to BT4 .pb.gz weights.")
    parser.add_argument("--out", required=True, help="Output policy_margin_report.json path.")
    parser.add_argument("--layer", default="trunk")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--vector", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-count", type=int, default=0)
    parser.add_argument("--control-kind", choices=["random", "shuffled"], default="random")
    parser.add_argument("--control-seed", type=int, default=0)
    args = parser.parse_args()

    if chess is None:
        raise ImportError("python-chess is required for dynamic policy-margin validation.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    rows = _load_pair_rows(Path(args.pairs), max_pairs=args.max_pairs, seed=args.seed)
    if not rows["root_fens"]:
        raise ValueError("No pair rows loaded for dynamic policy-margin validation.")

    best_indices = []
    subpar_indices = []
    valid_rows = {
        "root_fens": [],
        "root_history_fens": [],
        "best_moves": [],
        "subpar_moves": [],
    }
    legal_masks = []
    skipped = 0
    for root_fen, root_history, best_move, subpar_move in zip(
        rows["root_fens"],
        rows["root_history_fens"],
        rows["best_moves"],
        rows["subpar_moves"],
    ):
        try:
            best_idx = move_to_policy_index(str(best_move), "lc0_1858")
            subpar_idx = move_to_policy_index(str(subpar_move), "lc0_1858")
            board = chess.Board(str(root_fen))
            history_fens = normalize_history_fens(root_history, str(root_fen))
            for history_fen in history_fens:
                chess.Board(history_fen)
        except (KeyError, ValueError):
            skipped += 1
            continue
        if (
            chess.Move.from_uci(str(best_move)) not in board.legal_moves
            or chess.Move.from_uci(str(subpar_move)) not in board.legal_moves
        ):
            skipped += 1
            continue
        best_indices.append(best_idx)
        subpar_indices.append(subpar_idx)
        legal_masks.append(legal_move_mask(board, "lc0_1858"))
        valid_rows["root_fens"].append(str(root_fen))
        valid_rows["root_history_fens"].append(history_fens)
        valid_rows["best_moves"].append(str(best_move))
        valid_rows["subpar_moves"].append(str(subpar_move))
    if not valid_rows["root_fens"]:
        raise ValueError("No valid pair rows remained after move/FEN validation.")

    bundle = load_pb_gz(args.pb)
    params = map_bt4_weights(bundle, mapping_table=attention_policy_map())
    direction = _load_direction(Path(args.concept), vector=args.vector)

    plane_batches = _encode_plane_batches(valid_rows, batch_size=args.batch_size)
    base_policy = _forward_policy_batches(
        params,
        plane_batches,
        layer=args.layer,
        direction=None,
        alpha=args.alpha,
    )
    patched_policy = _forward_policy_batches(
        params,
        plane_batches,
        layer=args.layer,
        direction=direction,
        alpha=args.alpha,
    )

    report = policy_margin_report(
        base_policy=base_policy,
        patched_policy=patched_policy,
        best_indices=best_indices,
        subpar_indices=subpar_indices,
        legal_masks=np.asarray(legal_masks, dtype=bool),
        root_fens=valid_rows["root_fens"],
        best_moves=valid_rows["best_moves"],
        subpar_moves=valid_rows["subpar_moves"],
    )
    report.update(
        {
            "pairs": args.pairs,
            "concept": args.concept,
            "pb": args.pb,
            "layer": args.layer,
            "alpha": args.alpha,
            "vector": args.vector,
            "skipped_rows": int(skipped),
            "control_kind": args.control_kind,
            "control_count": args.control_count,
            "control_seed": args.control_seed,
        }
    )
    control_reports = []
    for control_kind, control_idx, control_direction in _control_directions(
        direction,
        kind=args.control_kind,
        count=args.control_count,
        seed=args.control_seed,
    ):
        control_policy = _forward_policy_batches(
            params,
            plane_batches,
            layer=args.layer,
            direction=control_direction,
            alpha=args.alpha,
        )
        control_report = policy_margin_report(
            base_policy=base_policy,
            patched_policy=control_policy,
            best_indices=best_indices,
            subpar_indices=subpar_indices,
            legal_masks=np.asarray(legal_masks, dtype=bool),
            root_fens=valid_rows["root_fens"],
            best_moves=valid_rows["best_moves"],
            subpar_moves=valid_rows["subpar_moves"],
        )
        control_report.pop("examples", None)
        control_report.update(
            {"control_kind": control_kind, "control_index": int(control_idx)}
        )
        control_reports.append(control_report)
    if control_reports:
        report["controls"] = control_reports
        report["control_calibration"] = control_calibration_report(
            report,
            control_reports,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dynamic policy-margin report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
