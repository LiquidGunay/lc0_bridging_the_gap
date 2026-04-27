"""Solve sparse concept vectors from stored optimal-vs-subpar rollout pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lc0jax.interpretability.concepts import (
    dynamic_rollout_differences,
    solve_screened_sparse_concept_from_differences,
    solve_sparse_concept_from_differences,
)


def _load_differences(path: Path, *, mode: str, index_mode: str, reverse: bool) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "differences" in data:
        differences = data["differences"]
    elif "optimal_rollouts" in data and "subpar_rollouts" in data:
        differences = dynamic_rollout_differences(
            data["optimal_rollouts"],
            data["subpar_rollouts"],
            mode=mode,
            index_mode=index_mode,
        )
    else:
        raise KeyError(
            "Pair file must contain either 'differences' or "
            "'optimal_rollouts' plus 'subpar_rollouts'"
        )
    differences = np.asarray(differences)
    if differences.ndim != 2:
        raise ValueError(f"Expected rank-2 differences after loading, got {differences.shape}")
    return -differences if reverse else differences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        required=True,
        help="NPZ file with differences or rollout activations.",
    )
    parser.add_argument("--out", required=True, help="Output concept directory.")
    parser.add_argument("--mode", choices=["flat", "mean"], default="flat")
    parser.add_argument(
        "--index-mode",
        choices=["both", "single_even", "single_odd"],
        default="both",
        help="Which rollout plies to aggregate before differencing.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Solve the prophylactic reversed-sign objective.",
    )
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help=(
            "Optional deterministic feature-screening cap before the CVXPY solve. "
            "The solved direction is expanded back to the original dimension."
        ),
    )
    parser.add_argument(
        "--screening-method",
        choices=["abs_mean", "mean_abs"],
        default="abs_mean",
        help="Feature scoring method used when --max-features is set.",
    )
    args = parser.parse_args()

    differences = _load_differences(
        Path(args.pairs),
        mode=args.mode,
        index_mode=args.index_mode,
        reverse=args.reverse,
    )
    if args.max_features is None:
        result = solve_sparse_concept_from_differences(
            differences,
            c=args.c,
            margin=args.margin,
            standardize=not args.no_standardize,
        )
    else:
        result = solve_screened_sparse_concept_from_differences(
            differences,
            max_features=args.max_features,
            screening_method=args.screening_method,
            c=args.c,
            margin=args.margin,
            standardize=not args.no_standardize,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    direction_payload = {
        "direction": result["direction"],
        "raw_direction": result["raw_direction"],
        "standardized_direction": result["standardized_direction"],
        "standardized_raw_direction": result["standardized_raw_direction"],
        "feature_std": result["feature_std"],
    }
    if "screening_indices" in result:
        direction_payload["screening_indices"] = result["screening_indices"]
        direction_payload["screening_scores"] = result["screening_scores"]
    np.savez_compressed(out_dir / "concept_direction.npz", **direction_payload)
    screening_enabled = bool(result.get("screening_enabled", False))
    screening_report = {
        "enabled": screening_enabled,
        "method": result.get("screening_method"),
        "max_features": result.get("screening_max_features"),
        "source_dimension": result.get("source_dimension", int(differences.shape[1])),
        "screened_dimension": result.get("screened_dimension", int(differences.shape[1])),
    }
    if "screening_indices" in result:
        screening_report["indices_stored_in"] = "concept_direction.npz:screening_indices"
        screening_report["score_key"] = "concept_direction.npz:screening_scores"
        screening_report["selected_feature_preview"] = [
            int(value) for value in result["screening_indices"][:20]
        ]
    report = {
        "method": "dynamic_screened_sparse_cvxpy" if screening_enabled else "dynamic_sparse_cvxpy",
        "pairs": str(args.pairs),
        "mode": args.mode,
        "index_mode": args.index_mode,
        "reverse": args.reverse,
        "num_pairs": int(differences.shape[0]),
        "dimension": int(differences.shape[1]),
        "c": args.c,
        "margin": args.margin,
        "standardize": not args.no_standardize,
        "screening": screening_report,
        "norm": result["norm"],
        "constraint_satisfaction": result["constraint_satisfaction"],
        "margin_satisfaction": result["margin_satisfaction"],
        "objective": result["objective"],
        "status": result["status"],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dynamic concept written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
