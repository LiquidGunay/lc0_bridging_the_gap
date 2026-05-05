"""Sweep screened dynamic concept solver settings and summarize reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lc0jax.interpretability.concepts import solve_screened_sparse_concept_from_differences
from lc0jax.interpretability.dynamic_baselines import dynamic_baseline_report
from lc0jax.interpretability.dynamic_causal import policy_margin_report
from lc0jax.interpretability.dynamic_evaluation import dynamic_evaluation_report
from lc0jax.interpretability.dynamic_prototypes import dynamic_prototype_report
from lc0jax.interpretability.dynamic_teachability import teachability_curriculum_records
from lc0jax.interpretability.pair_builders import normalize_history_fens

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None


def _parse_int_list(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("At least one integer value is required")
    if any(value < 1 for value in values):
        raise ValueError("Integer list values must be >= 1")
    return values


def _parse_float_list(text: str) -> list[float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("At least one float value is required")
    return values


def _parse_str_list(text: str) -> list[str]:
    values = [token.strip() for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("At least one value is required")
    return values


def _load_npz_payload(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_differences(path: str | Path, *, reverse: bool = False) -> np.ndarray:
    payload = _load_npz_payload(path)
    if "differences" not in payload:
        raise KeyError("pairs.npz must contain materialized 'differences'")
    differences = np.asarray(payload["differences"])
    return -differences if reverse else differences


def _json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_concept_artifacts(
    concept_dir: Path,
    *,
    result: dict[str, Any],
    train_pairs: Path,
    mode: str,
    index_mode: str,
    reverse: bool,
    num_pairs: int,
    dimension: int,
    c: float,
    margin: float,
    standardize: bool,
) -> dict[str, Any]:
    concept_dir.mkdir(parents=True, exist_ok=True)
    direction_payload = {
        "direction": result["direction"],
        "raw_direction": result["raw_direction"],
        "standardized_direction": result["standardized_direction"],
        "standardized_raw_direction": result["standardized_raw_direction"],
        "feature_std": result["feature_std"],
        "screening_indices": result["screening_indices"],
        "screening_scores": result["screening_scores"],
    }
    np.savez_compressed(concept_dir / "concept_direction.npz", **direction_payload)
    screening_report = {
        "enabled": bool(result.get("screening_enabled", False)),
        "method": result.get("screening_method"),
        "max_features": result.get("screening_max_features"),
        "source_dimension": result.get("source_dimension", dimension),
        "screened_dimension": result.get("screened_dimension", dimension),
        "indices_stored_in": "concept_direction.npz:screening_indices",
        "score_key": "concept_direction.npz:screening_scores",
        "selected_feature_preview": [
            int(value) for value in result["screening_indices"][:20]
        ],
    }
    report = {
        "method": "dynamic_screened_sparse_cvxpy",
        "pairs": str(train_pairs),
        "mode": mode,
        "index_mode": index_mode,
        "reverse": bool(reverse),
        "num_pairs": int(num_pairs),
        "dimension": int(dimension),
        "c": float(c),
        "margin": float(margin),
        "standardize": bool(standardize),
        "screening": screening_report,
        "norm": result["norm"],
        "constraint_satisfaction": result["constraint_satisfaction"],
        "margin_satisfaction": result["margin_satisfaction"],
        "objective": result["objective"],
        "status": result["status"],
    }
    _json_dump(concept_dir / "report.json", report)
    return report


def _load_direction(concept_dir: Path, *, key: str) -> np.ndarray:
    data = np.load(concept_dir / "concept_direction.npz", allow_pickle=True)
    return np.asarray(data[key])


def _load_pair_rows(
    path: Path,
    *,
    max_pairs: int | None,
    seed: int,
) -> tuple[dict[str, list[Any]], list[int]]:
    data = np.load(path, allow_pickle=True)
    required = ["root_fens", "best_moves", "subpar_moves"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"pairs.npz missing required metadata keys: {missing}")
    rows = {key: data[key].tolist() for key in required}
    if "root_history_fens" in data:
        rows["root_history_fens"] = data["root_history_fens"].tolist()
    else:
        rows["root_history_fens"] = [[fen] for fen in rows["root_fens"]]
    count = min(len(rows["root_fens"]), len(rows["best_moves"]), len(rows["subpar_moves"]))
    indices = np.arange(count)
    if max_pairs is not None and count > max_pairs:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_pairs, replace=False))
    return {key: [rows[key][idx] for idx in indices] for key in rows}, [
        int(idx) for idx in indices
    ]


def _policy_rows(path: Path, *, max_pairs: int | None, seed: int) -> dict[str, Any] | None:
    if chess is None:
        raise ImportError("python-chess is required for dynamic policy-margin validation.")
    from lc0jax.modeling.encode import encode_board
    from lc0jax.modeling.policy import legal_move_mask, move_to_policy_index

    rows, sampled_indices = _load_pair_rows(path, max_pairs=max_pairs, seed=seed)
    best_indices = []
    subpar_indices = []
    legal_masks = []
    planes = []
    valid_rows = {
        "root_fens": [],
        "root_history_fens": [],
        "best_moves": [],
        "subpar_moves": [],
    }
    valid_indices = []
    skipped = 0
    for sampled_idx, root_fen, root_history, best_move, subpar_move in zip(
        sampled_indices,
        rows["root_fens"],
        rows["root_history_fens"],
        rows["best_moves"],
        rows["subpar_moves"],
    ):
        try:
            board = chess.Board(str(root_fen))
            history_fens = normalize_history_fens(root_history, str(root_fen))
            history_boards = [chess.Board(fen) for fen in history_fens]
            best_idx = move_to_policy_index(str(best_move), "lc0_1858")
            subpar_idx = move_to_policy_index(str(subpar_move), "lc0_1858")
            best_chess_move = chess.Move.from_uci(str(best_move))
            subpar_chess_move = chess.Move.from_uci(str(subpar_move))
        except (KeyError, ValueError):
            skipped += 1
            continue
        if best_chess_move not in board.legal_moves or subpar_chess_move not in board.legal_moves:
            skipped += 1
            continue
        planes.append(
            encode_board(
                board,
                history_boards,
                planes_layout="nchw",
                input_format="INPUT_CLASSICAL_112_PLANE",
            )
        )
        best_indices.append(best_idx)
        subpar_indices.append(subpar_idx)
        legal_masks.append(legal_move_mask(board, "lc0_1858"))
        valid_rows["root_fens"].append(str(root_fen))
        valid_rows["root_history_fens"].append(history_fens)
        valid_rows["best_moves"].append(str(best_move))
        valid_rows["subpar_moves"].append(str(subpar_move))
        valid_indices.append(int(sampled_idx))
    if not planes:
        return None
    return {
        **valid_rows,
        "planes": np.stack(planes, axis=0),
        "best_indices": best_indices,
        "subpar_indices": subpar_indices,
        "legal_masks": np.asarray(legal_masks, dtype=bool),
        "sampled_indices": sampled_indices,
        "valid_indices": valid_indices,
        "skipped_rows": int(skipped),
    }


def _batch_indices(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def _base_policy_batches(params: dict, planes: np.ndarray, *, batch_size: int) -> np.ndarray:
    from lc0jax.modeling.inference import forward

    batches = []
    for start, stop in _batch_indices(planes.shape[0], batch_size):
        base_policy, _, _ = forward(params, planes[start:stop])
        batches.append(np.asarray(base_policy))
    return np.concatenate(batches, axis=0)


def _patched_policy_batches(
    params: dict,
    planes: np.ndarray,
    *,
    direction: np.ndarray,
    alpha: float,
    layer: str,
    batch_size: int,
) -> np.ndarray:
    from lc0jax.modeling.inference import forward

    batches = []
    for start, stop in _batch_indices(planes.shape[0], batch_size):
        patched_policy, _, _ = forward(
            params,
            planes[start:stop],
            patch={"layer": layer, "vector": direction, "alpha": alpha},
        )
        batches.append(np.asarray(patched_policy))
    return np.concatenate(batches, axis=0)


def _policy_margin_reports(
    *,
    params: dict | None,
    rows: dict[str, Any] | None,
    concept_dir: Path,
    direction_keys: list[str],
    alphas: list[float],
    layer: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    if params is None or rows is None:
        return []
    base_policy = _base_policy_batches(params, rows["planes"], batch_size=batch_size)
    reports = []
    for direction_key in direction_keys:
        direction = _load_direction(concept_dir, key=direction_key)
        for alpha in alphas:
            patched_policy = _patched_policy_batches(
                params,
                rows["planes"],
                direction=direction,
                alpha=alpha,
                layer=layer,
                batch_size=batch_size,
            )
            report = policy_margin_report(
                base_policy=base_policy,
                patched_policy=patched_policy,
                best_indices=rows["best_indices"],
                subpar_indices=rows["subpar_indices"],
                legal_masks=rows["legal_masks"],
                root_fens=rows["root_fens"],
                best_moves=rows["best_moves"],
                subpar_moves=rows["subpar_moves"],
            )
            report.update(
                {
                    "direction_key": direction_key,
                    "layer": layer,
                    "alpha": float(alpha),
                    "policy_rows": {
                        "sampled_indices": rows["sampled_indices"],
                        "valid_indices": rows["valid_indices"],
                        "sampled_count": len(rows["sampled_indices"]),
                        "valid_count": len(rows["valid_indices"]),
                    },
                    "skipped_rows": rows["skipped_rows"],
                }
            )
            reports.append(report)
    return reports


def _slug_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _config_id(method: str, max_features: int) -> str:
    return f"{method}_{max_features}"


def _row_from_reports(
    *,
    config_id: str,
    method: str,
    max_features: int,
    solve_report: dict[str, Any],
    eval_report: dict[str, Any],
    baselines_report: dict[str, Any],
    prototypes_report: dict[str, Any],
    curriculum_count: int,
    policy_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluation = eval_report["evaluation"]
    random_sparse = baselines_report.get("random_sparse", {})
    shuffled = baselines_report.get("shuffled_labels", {})
    largest_abs_policy = {}
    best_positive_policy = {}
    if policy_reports:
        largest_abs_policy = max(
            policy_reports,
            key=lambda item: abs(item["mean_delta_margin"]),
        )
        positive_reports = [
            item for item in policy_reports if item["mean_delta_margin"] > 0.0
        ]
        if positive_reports:
            best_positive_policy = max(
                positive_reports,
                key=lambda item: item["mean_delta_margin"],
            )
    return {
        "config_id": config_id,
        "screening_method": method,
        "max_features": int(max_features),
        "solve_status": solve_report["status"],
        "train_constraint_satisfaction": solve_report["constraint_satisfaction"],
        "train_margin_satisfaction": solve_report["margin_satisfaction"],
        "heldout_constraint_satisfaction": evaluation["constraint_satisfaction"],
        "heldout_margin_satisfaction": evaluation["margin_satisfaction"],
        "heldout_mean_score": evaluation["mean_score"],
        "random_constraint_satisfaction_mean": random_sparse.get(
            "constraint_satisfaction_mean"
        ),
        "shuffled_constraint_satisfaction_mean": shuffled.get(
            "constraint_satisfaction_mean"
        ),
        "prototype_top_k": prototypes_report["top_k"],
        "curriculum_lines": int(curriculum_count),
        "largest_abs_policy_direction_key": largest_abs_policy.get("direction_key"),
        "largest_abs_policy_alpha": largest_abs_policy.get("alpha"),
        "largest_abs_policy_mean_delta_margin": largest_abs_policy.get(
            "mean_delta_margin"
        ),
        "largest_abs_policy_top1_change_rate": largest_abs_policy.get(
            "top1_change_rate"
        ),
        "best_positive_policy_direction_key": best_positive_policy.get("direction_key"),
        "best_positive_policy_alpha": best_positive_policy.get("alpha"),
        "best_positive_policy_mean_delta_margin": best_positive_policy.get(
            "mean_delta_margin"
        ),
        "best_positive_policy_top1_change_rate": best_positive_policy.get(
            "top1_change_rate"
        ),
    }


def _format_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _markdown_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Dynamic Screening Sweep",
        "",
        (
            "| config | heldout constraint | heldout margin | random constraint | "
            "policy delta | policy alpha |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {config} | {constraint} | {margin} | {random} | {delta} | {alpha} |".format(
                config=row["config_id"],
                constraint=_format_cell(row["heldout_constraint_satisfaction"]),
                margin=_format_cell(row["heldout_margin_satisfaction"]),
                random=_format_cell(row["random_constraint_satisfaction_mean"]),
                delta=_format_cell(row["largest_abs_policy_mean_delta_margin"]),
                alpha=_format_cell(row["largest_abs_policy_alpha"]),
            )
        )
    lines.extend(
        [
            "",
            (
                "Policy delta is the largest absolute mean best-vs-subpar margin "
                "change among requested alpha/direction-key settings."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pairs = Path(args.train_pairs)
    test_pairs = Path(args.test_pairs)
    train_differences = _load_differences(train_pairs, reverse=args.reverse)
    test_differences = _load_differences(test_pairs, reverse=args.reverse)
    train_payload = _load_npz_payload(train_pairs)

    policy_params = None
    policy_rows = None
    alphas = _parse_float_list(args.policy_alphas)
    if args.pb and not args.skip_policy_margin:
        from lc0jax.modeling.policy import attention_policy_map
        from lc0jax.modeling.weights import load_pb_gz, map_bt4_weights

        policy_rows = _policy_rows(
            test_pairs,
            max_pairs=args.policy_max_pairs,
            seed=args.seed,
        )
        if policy_rows is None:
            raise ValueError("No valid pair rows remained for policy-margin sweep.")
        bundle = load_pb_gz(args.pb)
        policy_params = map_bt4_weights(bundle, mapping_table=attention_policy_map())

    rows = []
    configs = []
    methods = _parse_str_list(args.screening_methods)
    max_features_list = _parse_int_list(args.max_features)
    for method in methods:
        if method not in {"abs_mean", "mean_abs"}:
            raise ValueError(f"Unsupported screening method: {method}")
        for max_features in max_features_list:
            config = _config_id(method, max_features)
            concept_dir = out_dir / config
            result = solve_screened_sparse_concept_from_differences(
                train_differences,
                max_features=max_features,
                screening_method=method,
                c=args.c,
                margin=args.margin,
                standardize=not args.no_standardize,
            )
            solve_report = _write_concept_artifacts(
                concept_dir,
                result=result,
                train_pairs=train_pairs,
                mode=args.mode,
                index_mode=args.index_mode,
                reverse=args.reverse,
                num_pairs=int(train_differences.shape[0]),
                dimension=int(train_differences.shape[1]),
                c=args.c,
                margin=args.margin,
                standardize=not args.no_standardize,
            )
            eval_direction = _load_direction(concept_dir, key=args.evaluation_direction_key)
            eval_report = dynamic_evaluation_report(
                test_differences,
                eval_direction,
                margin=args.margin,
                split_name=args.split_name,
                direction_key=args.evaluation_direction_key,
            )
            eval_report["pairs"] = str(test_pairs)
            eval_report["concept"] = str(concept_dir)
            _json_dump(concept_dir / "heldout_eval_report.json", eval_report)

            baseline_direction = _load_direction(concept_dir, key=args.baseline_direction_key)
            baselines_report = dynamic_baseline_report(
                test_differences,
                baseline_direction,
                margin=args.margin,
                random_count=args.random_count,
                shuffled_label_count=args.shuffled_label_count,
                shuffled_solve_count=args.shuffled_solve_count,
                seed=args.seed,
                c=args.c,
                standardize=not args.no_standardize,
            )
            baselines_report["pairs"] = str(test_pairs)
            baselines_report["concept"] = str(concept_dir)
            _json_dump(concept_dir / "baselines_report.json", baselines_report)

            prototype_direction = _load_direction(concept_dir, key=args.prototype_direction_key)
            prototypes_report = dynamic_prototype_report(
                train_differences,
                prototype_direction,
                train_payload,
                top_k=args.top_k,
                random_count=args.prototype_random_count,
                seed=args.seed,
                split_name="train",
                direction_key=args.prototype_direction_key,
                reverse=args.reverse,
            )
            prototypes_report["pairs"] = str(train_pairs)
            prototypes_report["concept"] = str(concept_dir)
            prototypes_report["score_mode"] = "reverse" if args.reverse else "forward"
            _json_dump(concept_dir / "prototypes_report.json", prototypes_report)

            curriculum_rows = teachability_curriculum_records(prototypes_report)
            _write_jsonl(concept_dir / "teachability_curriculum.jsonl", curriculum_rows)

            policy_reports = _policy_margin_reports(
                params=policy_params,
                rows=policy_rows,
                concept_dir=concept_dir,
                direction_keys=_parse_str_list(args.policy_direction_keys),
                alphas=alphas,
                layer=args.policy_layer,
                batch_size=args.policy_batch_size,
            )
            for report in policy_reports:
                direction_key = str(report["direction_key"])
                alpha_slug = _slug_float(float(report["alpha"]))
                report.update(
                    {
                        "pb": args.pb,
                        "policy_max_pairs": args.policy_max_pairs,
                        "policy_batch_size": args.policy_batch_size,
                        "seed": args.seed,
                    }
                )
                _json_dump(
                    concept_dir / f"policy_margin_{direction_key}_alpha_{alpha_slug}.json",
                    report,
                )
            if policy_reports:
                _json_dump(concept_dir / "policy_margin_reports.json", {"runs": policy_reports})

            configs.append(
                {
                    "config_id": config,
                    "concept_dir": str(concept_dir),
                    "solve_report": str(concept_dir / "report.json"),
                    "heldout_eval_report": str(concept_dir / "heldout_eval_report.json"),
                    "baselines_report": str(concept_dir / "baselines_report.json"),
                    "prototypes_report": str(concept_dir / "prototypes_report.json"),
                    "curriculum": str(concept_dir / "teachability_curriculum.jsonl"),
                    "policy_margin_reports": [
                        str(
                            concept_dir
                            / (
                                "policy_margin_"
                                f"{item['direction_key']}_alpha_"
                                f"{_slug_float(float(item['alpha']))}.json"
                            )
                        )
                        for item in policy_reports
                    ],
                }
            )
            rows.append(
                _row_from_reports(
                    config_id=config,
                    method=method,
                    max_features=max_features,
                    solve_report=solve_report,
                    eval_report=eval_report,
                    baselines_report=baselines_report,
                    prototypes_report=prototypes_report,
                    curriculum_count=len(curriculum_rows),
                    policy_reports=policy_reports,
                )
            )

    summary = {
        "method": "dynamic_screening_sweep",
        "train_pairs": str(train_pairs),
        "test_pairs": str(test_pairs),
        "out": str(out_dir),
        "mode": args.mode,
        "index_mode": args.index_mode,
        "reverse": bool(args.reverse),
        "margin": float(args.margin),
        "c": float(args.c),
        "standardize": not args.no_standardize,
        "screening_methods": methods,
        "max_features": max_features_list,
        "policy_alphas": [] if args.skip_policy_margin or not args.pb else alphas,
        "policy_direction_keys": _parse_str_list(args.policy_direction_keys),
        "policy": {
            "enabled": bool(args.pb and not args.skip_policy_margin),
            "pb": args.pb,
            "layer": args.policy_layer,
            "max_pairs": args.policy_max_pairs,
            "batch_size": args.policy_batch_size,
            "seed": args.seed,
            "sampled_indices": [] if policy_rows is None else policy_rows["sampled_indices"],
            "valid_indices": [] if policy_rows is None else policy_rows["valid_indices"],
            "skipped_rows": None if policy_rows is None else policy_rows["skipped_rows"],
        },
        "rows": rows,
        "configs": configs,
    }
    _json_dump(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(_markdown_summary(rows), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--test-pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["flat", "mean"], default="flat")
    parser.add_argument(
        "--index-mode",
        choices=["both", "single_even", "single_odd"],
        default="both",
    )
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--max-features", default="1024,2048,4096,8192")
    parser.add_argument("--screening-methods", default="abs_mean,mean_abs")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--evaluation-direction-key", default="raw_direction")
    parser.add_argument("--baseline-direction-key", default="direction")
    parser.add_argument("--prototype-direction-key", default="direction")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--random-count", type=int, default=128)
    parser.add_argument("--shuffled-label-count", type=int, default=128)
    parser.add_argument("--shuffled-solve-count", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--prototype-random-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pb", help="Optional BT4 .pb.gz path for policy-margin sweeps.")
    parser.add_argument("--skip-policy-margin", action="store_true")
    parser.add_argument("--policy-alphas", default="0.1,0.3,1.0,3.0")
    parser.add_argument("--policy-direction-keys", default="direction")
    parser.add_argument("--policy-layer", default="trunk")
    parser.add_argument("--policy-batch-size", type=int, default=4)
    parser.add_argument("--policy-max-pairs", type=int, default=16)
    args = parser.parse_args()

    run_sweep(args)
    print(f"Dynamic screening sweep written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
