"""Evaluate a dynamic concept direction on a held-out pair split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc0jax.interpretability.dynamic_artifacts import (
    load_concept_direction,
    load_pair_differences,
)
from lc0jax.interpretability.dynamic_evaluation import dynamic_evaluation_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Held-out solver-ready pairs.npz file.")
    parser.add_argument("--concept", required=True, help="Concept directory or direction npz.")
    parser.add_argument("--out", required=True, help="Output heldout_eval_report.json path.")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--split-name", default="heldout")
    parser.add_argument(
        "--direction-key",
        default="raw_direction",
        help="Direction array to evaluate from concept_direction.npz.",
    )
    args = parser.parse_args()

    direction = load_concept_direction(args.concept, key=args.direction_key)
    report = dynamic_evaluation_report(
        load_pair_differences(args.pairs),
        direction,
        margin=args.margin,
        split_name=args.split_name,
        direction_key=args.direction_key,
    )
    report["pairs"] = args.pairs
    report["concept"] = args.concept

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dynamic concept evaluation written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
