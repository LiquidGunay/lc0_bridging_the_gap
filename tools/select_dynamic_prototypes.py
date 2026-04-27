"""Select dynamic concept prototypes and random controls from pair rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lc0jax.interpretability.dynamic_artifacts import (
    load_concept_direction,
    load_pair_differences,
)
from lc0jax.interpretability.dynamic_prototypes import dynamic_prototype_report


def _load_pair_metadata(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Solver-ready pairs.npz file.")
    parser.add_argument("--concept", required=True, help="Concept directory or direction npz.")
    parser.add_argument("--out", required=True, help="Output prototypes_report.json path.")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--random-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-name", default="train")
    parser.add_argument(
        "--direction-key",
        default="direction",
        help="Direction array to use from concept_direction.npz.",
    )
    args = parser.parse_args()

    report = dynamic_prototype_report(
        load_pair_differences(args.pairs),
        load_concept_direction(args.concept, key=args.direction_key),
        _load_pair_metadata(args.pairs),
        top_k=args.top_k,
        random_count=args.random_count,
        seed=args.seed,
        split_name=args.split_name,
        direction_key=args.direction_key,
    )
    report["pairs"] = args.pairs
    report["concept"] = args.concept

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dynamic prototypes written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
