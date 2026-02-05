"""Filter FEN lists with simple heuristics."""

from __future__ import annotations

import argparse

from lc0jax.interpretability.datasets import filter_fens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens", required=True, help="Input FEN list (one per line).")
    parser.add_argument("--out", required=True, help="Output FEN list.")
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--min-ply", type=int, default=None)
    parser.add_argument("--max-ply", type=int, default=None)
    parser.add_argument("--min-phase", type=float, default=None)
    parser.add_argument("--max-phase", type=float, default=None)
    parser.add_argument("--min-pieces", type=int, default=None)
    parser.add_argument("--max-pieces", type=int, default=None)
    parser.add_argument("--min-nonpawn", type=int, default=None)
    parser.add_argument("--max-nonpawn", type=int, default=None)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument("--progress-label", default=None)
    args = parser.parse_args()

    kept = filter_fens(
        args.fens,
        out_fens=args.out,
        max_positions=args.max_positions,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        min_phase=args.min_phase,
        max_phase=args.max_phase,
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
        min_nonpawn=args.min_nonpawn,
        max_nonpawn=args.max_nonpawn,
        dedupe=args.dedupe,
        progress_every=args.progress_every,
        progress_label=args.progress_label,
    )
    print(f"Kept positions: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
