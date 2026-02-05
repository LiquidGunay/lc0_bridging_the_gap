"""Convert PGN to a newline-delimited FEN list."""

from __future__ import annotations

import argparse

from lc0jax.interpretability.datasets import pgn_to_fens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--ply-stride", type=int, default=1)
    args = parser.parse_args()

    pgn_to_fens(args.pgn, out_path=args.out, max_positions=args.max_positions, ply_stride=args.ply_stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
