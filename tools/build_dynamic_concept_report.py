"""Build a markdown report card for a dynamic rollout concept run."""

from __future__ import annotations

import argparse
from pathlib import Path

from lc0jax.interpretability.dynamic_reports import build_dynamic_concept_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Solver-ready pairs.npz file.")
    parser.add_argument("--concept", required=True, help="Dynamic concept output directory.")
    parser.add_argument(
        "--novelty",
        help="Optional novelty_report.json path. Defaults to <concept>/novelty_report.json.",
    )
    parser.add_argument(
        "--baselines",
        help="Optional baselines_report.json path. Defaults to <concept>/baselines_report.json.",
    )
    parser.add_argument(
        "--evaluation",
        help=(
            "Optional heldout_eval_report.json path. Defaults to "
            "<concept>/heldout_eval_report.json."
        ),
    )
    parser.add_argument(
        "--policy-margin",
        help=(
            "Optional policy_margin_report.json path. Defaults to "
            "<concept>/policy_margin_report.json."
        ),
    )
    parser.add_argument("--out", required=True, help="Output markdown path.")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    report = build_dynamic_concept_report(
        pairs_path=args.pairs,
        concept_dir=args.concept,
        novelty_path=args.novelty,
        evaluation_path=args.evaluation,
        baselines_path=args.baselines,
        policy_margin_path=args.policy_margin,
        top_n=args.top_n,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Dynamic concept report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
