"""Filter PGN games by rating and time control, optionally producing FENs."""

from __future__ import annotations

import argparse

from lc0jax.interpretability.datasets import filter_pgn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--out-pgn", default=None)
    parser.add_argument("--out-fens", default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--ply-stride", type=int, default=1)
    parser.add_argument("--min-elo", type=int, default=None)
    parser.add_argument(
        "--time-class",
        action="append",
        default=None,
        help="Time control class (repeatable or comma-separated)",
    )
    parser.add_argument("--rated", action="store_true")
    parser.add_argument("--casual", action="store_true")
    args = parser.parse_args()

    require_rated = None
    if args.rated and args.casual:
        raise SystemExit("Choose only one of --rated or --casual")
    if args.rated:
        require_rated = True
    if args.casual:
        require_rated = False

    time_class = None
    if args.time_class:
        allowed = {"ultrabullet", "bullet", "blitz", "rapid", "classical"}
        merged: list[str] = []
        for item in args.time_class:
            if item is None:
                continue
            merged.extend([part.strip() for part in item.split(",") if part.strip()])
        invalid = [item for item in merged if item not in allowed]
        if invalid:
            raise SystemExit(f"Invalid time-class entries: {invalid}")
        time_class = merged

    kept = filter_pgn(
        args.pgn,
        out_pgn=args.out_pgn,
        out_fens=args.out_fens,
        max_games=args.max_games,
        ply_stride=args.ply_stride,
        min_elo=args.min_elo,
        time_class=time_class,
        require_rated=require_rated,
        require_standard=True,
    )
    print(f"Kept games: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
