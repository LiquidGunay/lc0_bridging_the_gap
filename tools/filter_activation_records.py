"""Filter JSONL activation records to a selected FEN list."""

from __future__ import annotations

import argparse

from lc0jax.interpretability.datasets import filter_activation_records_by_fens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--fens", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    kept = filter_activation_records_by_fens(
        args.records,
        fens_path=args.fens,
        out_path=args.out,
    )
    print(f"Kept {kept} activation records in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
