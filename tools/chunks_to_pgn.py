"""Convert LC0 training chunks to one-ply PGNs and/or FEN lists."""

from __future__ import annotations

import argparse
import os

import chess.pgn

from lc0jax.training.chunks import iter_records, record_to_board, record_to_move


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", action="append", help="Chunk file path (repeatable)")
    parser.add_argument("--chunk-dir", default=None, help="Directory to scan for .gz chunks (recursive)")
    parser.add_argument("--chunk-list", default=None, help="Text file with one chunk path per line")
    parser.add_argument("--out-pgn", default=None)
    parser.add_argument("--out-fens", default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--use-best", action="store_true")
    args = parser.parse_args()

    chunk_paths = []
    if args.chunk:
        chunk_paths.extend(args.chunk)
    if args.chunk_list:
        with open(args.chunk_list, "r", encoding="utf-8") as handle:
            for line in handle:
                path = line.strip()
                if path:
                    chunk_paths.append(path)
    if args.chunk_dir:
        for root, _dirs, files in os.walk(args.chunk_dir):
            for name in files:
                if name.endswith(".gz"):
                    chunk_paths.append(os.path.join(root, name))

    if not chunk_paths:
        raise SystemExit("No chunk files provided. Use --chunk, --chunk-dir, or --chunk-list.")

    # Deduplicate while keeping order.
    seen = set()
    unique_chunks = []
    for path in chunk_paths:
        if path in seen:
            continue
        seen.add(path)
        unique_chunks.append(path)

    out_pgn_file = open(args.out_pgn, "w", encoding="utf-8") if args.out_pgn else None
    out_fens_file = open(args.out_fens, "w", encoding="utf-8") if args.out_fens else None
    exporter = chess.pgn.FileExporter(out_pgn_file) if out_pgn_file else None

    count = 0
    try:
        for chunk_path in unique_chunks:
            for record in iter_records(chunk_path):
                board = record_to_board(record)
                move = record_to_move(record, use_best=args.use_best)

                if out_fens_file:
                    out_fens_file.write(board.fen() + "\n")

                if exporter:
                    game = chess.pgn.Game()
                    game.setup(board)
                    game.headers["SetUp"] = "1"
                    game.headers["FEN"] = board.fen()
                    if move and move in board.legal_moves:
                        game.add_variation(move)
                    game.accept(exporter)

                count += 1
                if args.max_positions is not None and count >= args.max_positions:
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        if out_pgn_file:
            out_pgn_file.close()
        if out_fens_file:
            out_fens_file.close()

    print(f"Wrote positions: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
