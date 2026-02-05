"""Run a lightweight Elo bench between two UCI engines (typically LC0)."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from lc0jax.uci.elo import elo_from_score, score_ci


@dataclass
class MatchResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games


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


def load_start_fens(path: str | None) -> list[str]:
    if not path:
        return []
    fens = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            fen = line.strip()
            if fen:
                fens.append(fen)
    return fens


def play_game(
    engine_a: chess.engine.SimpleEngine,
    engine_b: chess.engine.SimpleEngine,
    *,
    limit: chess.engine.Limit,
    a_is_white: bool,
    start_fen: str | None,
    max_moves: int,
) -> float:
    board = chess.Board(start_fen) if start_fen else chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        engine = engine_a if board.turn == a_is_white else engine_b
        result = engine.play(board, limit)
        if result.move is None:
            break
        board.push(result.move)
        move_count += 1

    outcome = board.outcome()
    if outcome is None:
        return 0.5
    if outcome.winner is None:
        return 0.5
    if outcome.winner == a_is_white:
        return 1.0
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc0", required=True, help="Path to LC0 binary.")
    parser.add_argument("--weights-a", required=True, help="Weights for engine A (baseline).")
    parser.add_argument("--weights-b", required=True, help="Weights for engine B (candidate).")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--movetime-ms", type=int, default=100)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--backend-opts", default=None)
    parser.add_argument("--uci-timeout", type=float, default=60.0)
    parser.add_argument("--start-fens", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-moves", type=int, default=200)
    args = parser.parse_args()

    if args.nodes is None and args.movetime_ms is None:
        raise ValueError("Provide --nodes or --movetime-ms (or both).")

    limit = chess.engine.Limit(
        nodes=args.nodes,
        time=None if args.movetime_ms is None else args.movetime_ms / 1000.0,
    )

    engine_a = chess.engine.SimpleEngine.popen_uci([args.lc0], timeout=args.uci_timeout)
    engine_b = chess.engine.SimpleEngine.popen_uci([args.lc0], timeout=args.uci_timeout)
    try:
        configure_engine(
            engine_a,
            weights=args.weights_a,
            threads=args.threads,
            backend=args.backend,
            backend_opts=args.backend_opts,
        )
        configure_engine(
            engine_b,
            weights=args.weights_b,
            threads=args.threads,
            backend=args.backend,
            backend_opts=args.backend_opts,
        )

        fens = load_start_fens(args.start_fens)
        rng = random.Random(args.seed)
        results = MatchResult()

        for game_idx in range(args.games):
            a_is_white = (game_idx % 2 == 0)
            start_fen = rng.choice(fens) if fens else None
            score = play_game(
                engine_a,
                engine_b,
                limit=limit,
                a_is_white=a_is_white,
                start_fen=start_fen,
                max_moves=args.max_moves,
            )
            if score == 1.0:
                results.wins += 1
            elif score == 0.0:
                results.losses += 1
            else:
                results.draws += 1

            print(
                f"Game {game_idx + 1}/{args.games}: "
                f"A score={score:.1f} (W/L/D={results.wins}/{results.losses}/{results.draws})",
                flush=True,
            )

        score = results.score
        elo = elo_from_score(score)
        ci_low, ci_high = score_ci(score, results.games)
        elo_low = elo_from_score(ci_low)
        elo_high = elo_from_score(ci_high)

        print("")
        print(f"Score: {score:.3f} over {results.games} games (W/L/D={results.wins}/{results.losses}/{results.draws})")
        print(f"Elo diff (B over A): {elo:.1f}")
        print(f"95% score CI: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"95% Elo CI: [{elo_low:.1f}, {elo_high:.1f}]")

    finally:
        engine_a.quit()
        engine_b.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
