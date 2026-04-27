"""Export dynamic prototype selections as teachability curriculum JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc0jax.interpretability.dynamic_teachability import teachability_curriculum_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototypes", required=True, help="prototypes_report.json path.")
    parser.add_argument("--out", required=True, help="Output curriculum JSONL path.")
    parser.add_argument("--max-prototypes", type=int, default=None)
    parser.add_argument("--max-controls", type=int, default=None)
    args = parser.parse_args()

    report = json.loads(Path(args.prototypes).read_text(encoding="utf-8"))
    records = teachability_curriculum_records(
        report,
        max_prototypes=args.max_prototypes,
        max_controls=args.max_controls,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} teachability curriculum records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
