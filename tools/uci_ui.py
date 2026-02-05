"""Minimal text UI for playing against a UCI engine (LC0)."""

from __future__ import annotations

import argparse
from typing import Optional

import chess
import chess.engine
import chess.pgn


def parse_move(board: chess.Board, text: str) -> Optional[chess.Move]:
    text = text.strip()
    if not text:
        return None
    try:
        return board.parse_uci(text)
    except ValueError:
        pass
    try:
        return board.parse_san(text)
    except ValueError:
        return None


def configure_engine(
    engine: chess.engine.SimpleEngine,
    *,
    weights: str | None,
    threads: int | None,
    backend: str | None,
    backend_opts: str | None,
) -> None:
    options = {}
    if weights:
        options["WeightsFile"] = weights
    if threads:
        options["Threads"] = threads
    if backend:
        options["Backend"] = backend
    if backend_opts:
        options["BackendOptions"] = backend_opts
    if options:
        engine.configure(options)


def _last_move_san(board: chess.Board) -> Optional[str]:
    if not board.move_stack:
        return None
    move = board.pop()
    san = board.san(move)
    board.push(move)
    return san


def _moves_san(board: chess.Board, start_board: chess.Board, *, max_len: int = 12) -> str:
    temp = start_board.copy(stack=False)
    parts = []
    for move in board.move_stack:
        if temp.turn == chess.WHITE:
            parts.append(f"{temp.fullmove_number}.")
        parts.append(temp.san(move))
        temp.push(move)
    if len(parts) > max_len:
        parts = ["..."] + parts[-max_len:]
    return " ".join(parts)


def _format_score(info: dict, board: chess.Board) -> str:
    if "score" not in info:
        return ""
    score = info["score"].pov(board.turn)
    return str(score)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc0", required=True, help="Path to LC0 binary.")
    parser.add_argument("--weights", required=True, help="Path to LC0 weights.")
    parser.add_argument("--fen", default=None, help="Optional starting FEN.")
    parser.add_argument("--human-side", choices=["white", "black"], default="white")
    parser.add_argument("--movetime-ms", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--backend-opts", default=None)
    parser.add_argument("--uci-timeout", type=float, default=60.0)
    parser.add_argument("--analysis", action="store_true", help="Show engine eval before each move.")
    parser.add_argument("--analysis-movetime-ms", type=int, default=None)
    parser.add_argument("--analysis-nodes", type=int, default=None)
    parser.add_argument("--pgn-out", default=None)
    args = parser.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci([args.lc0], timeout=args.uci_timeout)
    try:
        configure_engine(
            engine,
            weights=args.weights,
            threads=args.threads,
            backend=args.backend,
            backend_opts=args.backend_opts,
        )
        start_board = chess.Board(args.fen) if args.fen else chess.Board()
        board = start_board.copy(stack=False)
        human_is_white = args.human_side == "white"
        if args.nodes is None and args.movetime_ms is None:
            raise ValueError("Provide --nodes or --movetime-ms.")
        limit = chess.engine.Limit(
            nodes=args.nodes,
            time=None if args.movetime_ms is None else args.movetime_ms / 1000.0,
        )
        analysis_limit = chess.engine.Limit(
            nodes=args.analysis_nodes,
            time=None
            if args.analysis_movetime_ms is None
            else args.analysis_movetime_ms / 1000.0,
        )

        print(
            "Type moves in UCI (e2e4) or SAN (e4). Commands: 'fen <FEN>', 'undo', 'moves', 'quit'."
        )

        while True:
            print("")
            print(board)
            last_move = _last_move_san(board)
            if last_move:
                print(f"Last move: {last_move}")
            print(f"Moves: {_moves_san(board, start_board)}")
            print(f"Side to move: {'white' if board.turn else 'black'}")
            if board.is_game_over():
                print(f"Game over: {board.result()} ({board.outcome()})")
                break
            if args.analysis:
                info = engine.analyse(board, analysis_limit)
                score_text = _format_score(info, board)
                pv = info.get("pv")
                if score_text:
                    print(f"Eval: {score_text}")
                if pv:
                    pv_san = []
                    pv_board = board.copy(stack=False)
                    for move in pv[:6]:
                        pv_san.append(pv_board.san(move))
                        pv_board.push(move)
                    print(f"PV: {' '.join(pv_san)}")

            if board.turn == human_is_white:
                text = input("Your move> ").strip()
                if text.lower() in {"quit", "exit"}:
                    break
                if text.lower().startswith("fen "):
                    fen = text[4:].strip()
                    start_board = chess.Board(fen)
                    board = start_board.copy(stack=False)
                    continue
                if text.lower() == "undo":
                    if board.move_stack:
                        board.pop()
                    continue
                if text.lower() == "moves":
                    print(_moves_san(board, start_board, max_len=24))
                    continue
                move = parse_move(board, text)
                if move is None or move not in board.legal_moves:
                    print("Invalid move.")
                    continue
                board.push(move)
            else:
                result = engine.play(board, limit)
                if result.move is None:
                    print("Engine returned no move.")
                    break
                board.push(result.move)
                print(f"Engine move: {result.move.uci()}")
    finally:
        if args.pgn_out:
            game = chess.pgn.Game.from_board(board)
            with open(args.pgn_out, "w", encoding="utf-8") as handle:
                handle.write(str(game) + "\n\n")
        engine.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
