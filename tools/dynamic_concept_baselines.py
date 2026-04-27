"""Build random and shuffled baselines for a dynamic concept direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lc0jax.interpretability.dynamic_artifacts import load_concept_direction
from lc0jax.interpretability.dynamic_baselines import dynamic_baseline_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Solver-ready pairs.npz file.")
    parser.add_argument("--concept", required=True, help="Concept directory or direction npz.")
    parser.add_argument("--out", required=True, help="Output baselines_report.json path.")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--random-count", type=int, default=128)
    parser.add_argument("--shuffled-label-count", type=int, default=128)
    parser.add_argument(
        "--shuffled-solve-count",
        type=int,
        default=0,
        help="Optional expensive CVXPY solves on sign-shuffled differences.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args()

    pair_data = np.load(args.pairs, allow_pickle=True)
    if "differences" not in pair_data:
        raise KeyError("pairs.npz must contain materialized 'differences'")
    report = dynamic_baseline_report(
        pair_data["differences"],
        load_concept_direction(Path(args.concept)),
        margin=args.margin,
        random_count=args.random_count,
        shuffled_label_count=args.shuffled_label_count,
        shuffled_solve_count=args.shuffled_solve_count,
        seed=args.seed,
        c=args.c,
        standardize=not args.no_standardize,
    )
    report["pairs"] = args.pairs
    report["concept"] = args.concept

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dynamic concept baselines written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
