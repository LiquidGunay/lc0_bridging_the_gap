"""Filter FENs based on LC0 MCTS evaluation and heuristics."""

from __future__ import annotations

import argparse
from typing import Iterable

import chess
import chess.engine

from lc0jax.interpretability.datasets import _fen_phase, _fen_ply


def iter_fens(path: str, *, start_line: int = 0) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx < start_line:
                continue
            fen = line.strip()
            if fen:
                yield fen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens", required=True, help="Input FEN list")
    parser.add_argument("--out", required=True, help="Output FEN list")
    parser.add_argument("--lc0", required=True, help="Path to LC0 binary")
    parser.add_argument("--weights", default=None, help="Path to LC0 network weights")

    # Engine limits
    parser.add_argument("--nodes", type=int, default=800, help="MCTS nodes per position")
    parser.add_argument("--movetime-ms", type=int, default=None, help="MCTS time per position in ms")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--backend-opts", default=None)
    parser.add_argument("--uci-timeout", type=float, default=60.0)

    # Evaluation limits
    parser.add_argument("--min-cp", type=int, default=None, help="Minimum centipawn eval (from White's POV)")
    parser.add_argument("--max-cp", type=int, default=None, help="Maximum centipawn eval (from White's POV)")
    parser.add_argument("--min-win-prob", type=float, default=None, help="Minimum win probability [0.0, 1.0]")
    parser.add_argument("--max-win-prob", type=float, default=None, help="Maximum win probability [0.0, 1.0]")

    # Heuristic limits
    parser.add_argument("--min-ply", type=int, default=None)
    parser.add_argument("--max-ply", type=int, default=None)
    parser.add_argument("--min-phase", type=float, default=None)
    parser.add_argument("--max-phase", type=float, default=None)
    parser.add_argument("--min-pieces", type=int, default=None)
    parser.add_argument("--max-pieces", type=int, default=None)
    parser.add_argument("--min-nonpawn", type=int, default=None)
    parser.add_argument("--max-nonpawn", type=int, default=None)

    # Progress
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--start-line", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if args.nodes is None and args.movetime_ms is None:
        raise ValueError("Provide --nodes or --movetime-ms (or both).")

    engine = chess.engine.SimpleEngine.popen_uci([args.lc0], timeout=args.uci_timeout)
    options = {}
    if args.weights:
        options["WeightsFile"] = args.weights
    if args.threads:
        options["Threads"] = args.threads
    if args.backend:
        options["Backend"] = args.backend
    if args.backend_opts:
        options["BackendOptions"] = args.backend_opts

    if options:
        engine.configure(options)

    limit = chess.engine.Limit(
        nodes=args.nodes,
        time=None if args.movetime_ms is None else args.movetime_ms / 1000.0,
    )

    seen = 0
    kept = 0
    out_mode = "a" if args.append else "w"

    try:
        with open(args.out, out_mode, encoding="utf-8") as out_f:
            for fen in iter_fens(args.fens, start_line=args.start_line):
                seen += 1
                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue

                # 1. Heuristic filtering (Fast)
                ply = _fen_ply(board)
                if args.min_ply is not None and ply < args.min_ply:
                    continue
                if args.max_ply is not None and ply > args.max_ply:
                    continue

                phase = _fen_phase(board)
                if args.min_phase is not None and phase < args.min_phase:
                    continue
                if args.max_phase is not None and phase > args.max_phase:
                    continue

                piece_count = len(board.piece_map())
                if args.min_pieces is not None and piece_count < args.min_pieces:
                    continue
                if args.max_pieces is not None and piece_count > args.max_pieces:
                    continue

                nonpawn = piece_count - len(board.pieces(chess.PAWN, chess.WHITE)) - len(
                    board.pieces(chess.PAWN, chess.BLACK)
                )
                if args.min_nonpawn is not None and nonpawn < args.min_nonpawn:
                    continue
                if args.max_nonpawn is not None and nonpawn > args.max_nonpawn:
                    continue

                # 2. Engine evaluation filtering (Slow)
                try:
                    info = engine.analyse(board, limit)
                except Exception:
                    continue

                score = info.get("score")
                if score is None:
                    continue

                # Get score from White's POV
                pov_score = score.pov(chess.WHITE)

                keep = True

                if args.min_cp is not None or args.max_cp is not None:
                    cp = pov_score.score(mate_score=10000)
                    if cp is not None:
                        if args.min_cp is not None and cp < args.min_cp:
                            keep = False
                        if args.max_cp is not None and cp > args.max_cp:
                            keep = False
                    else:
                        keep = False # E.g. forced mate but no cp value

                if args.min_win_prob is not None or args.max_win_prob is not None:
                    # python-chess 1.10+ supports wdl
                    wdl = pov_score.wdl()
                    if wdl is not None:
                        win_prob = wdl.expectation()
                        if args.min_win_prob is not None and win_prob < args.min_win_prob:
                            keep = False
                        if args.max_win_prob is not None and win_prob > args.max_win_prob:
                            keep = False
                    else:
                        keep = False

                if keep:
                    out_f.write(fen + "\n")
                    kept += 1

                if args.progress_every and seen % args.progress_every == 0:
                    print(f"Seen {seen} positions, kept {kept}", flush=True)

                if args.max_positions is not None and kept >= args.max_positions:
                    break

    finally:
        engine.quit()

    print(f"Eval filter kept: {kept} (seen {seen})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
