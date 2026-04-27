"""Convert PGN to JSONL activation records with rolling board history."""

from __future__ import annotations

import argparse

from lc0jax.interpretability.datasets import pgn_to_activation_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--ply-stride", type=int, default=1)
    parser.add_argument("--history-len", type=int, default=8)
    args = parser.parse_args()

    written = pgn_to_activation_records(
        args.pgn,
        out_path=args.out,
        max_positions=args.max_positions,
        ply_stride=args.ply_stride,
        history_len=args.history_len,
    )
    print(f"Wrote {written} activation records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
