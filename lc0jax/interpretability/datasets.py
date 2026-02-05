"""Dataset parsing utilities for PGN and FEN inputs."""

from __future__ import annotations

from typing import Iterable, Sequence

import math
import re

import chess
import chess.pgn


def pgn_to_fens(pgn_path: str, *, out_path: str, max_positions: int | None = None, ply_stride: int = 1) -> list[str]:
    """Parse PGN into a list of FENs and optionally write them to disk."""
    if ply_stride < 1:
        raise ValueError("ply_stride must be >= 1")

    fens: list[str] = []
    out_file = None
    if out_path:
        out_file = open(out_path, "w", encoding="utf-8")

    try:
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                board = game.board()
                ply_idx = 0
                for move in game.mainline_moves():
                    board.push(move)
                    if ply_idx % ply_stride == 0:
                        fen = board.fen()
                        if out_file:
                            out_file.write(fen + "\n")
                        fens.append(fen)
                        if max_positions is not None and len(fens) >= max_positions:
                            return fens
                    ply_idx += 1
    finally:
        if out_file:
            out_file.close()

    return fens


def iter_fens(path: str) -> Iterable[str]:
    """Yield FEN strings from a newline-delimited FEN file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fen = line.strip()
            if fen:
                yield fen


def _fen_ply(board: chess.Board) -> int:
    fullmove = board.fullmove_number
    return (fullmove - 1) * 2 + (1 if board.turn == chess.BLACK else 0)


def _fen_phase(board: chess.Board) -> float:
    phase = 0.0
    phase += 4 * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)))
    phase += 2 * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)))
    phase += len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
    phase += len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
    return min(max(phase / 24.0, 0.0), 1.0)


def filter_fens(
    fen_path: str,
    *,
    out_fens: str,
    max_positions: int | None = None,
    min_ply: int | None = None,
    max_ply: int | None = None,
    min_phase: float | None = None,
    max_phase: float | None = None,
    min_pieces: int | None = None,
    max_pieces: int | None = None,
    min_nonpawn: int | None = None,
    max_nonpawn: int | None = None,
    dedupe: bool = False,
    progress_every: int | None = None,
    progress_label: str | None = None,
) -> int:
    """Filter FENs by simple heuristics (ply, phase, piece counts)."""
    kept = 0
    seen = 0
    seen_fens: set[str] | None = set() if dedupe else None
    out_file = open(out_fens, "w", encoding="utf-8")

    try:
        with open(fen_path, "r", encoding="utf-8") as handle:
            for line in handle:
                fen = line.strip()
                if not fen:
                    continue
                seen += 1
                if seen_fens is not None:
                    if fen in seen_fens:
                        continue
                    seen_fens.add(fen)

                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue

                ply = _fen_ply(board)
                if min_ply is not None and ply < min_ply:
                    continue
                if max_ply is not None and ply > max_ply:
                    continue

                phase = _fen_phase(board)
                if min_phase is not None and phase < min_phase:
                    continue
                if max_phase is not None and phase > max_phase:
                    continue

                piece_count = len(board.piece_map())
                if min_pieces is not None and piece_count < min_pieces:
                    continue
                if max_pieces is not None and piece_count > max_pieces:
                    continue

                nonpawn = piece_count - len(board.pieces(chess.PAWN, chess.WHITE)) - len(
                    board.pieces(chess.PAWN, chess.BLACK)
                )
                if min_nonpawn is not None and nonpawn < min_nonpawn:
                    continue
                if max_nonpawn is not None and nonpawn > max_nonpawn:
                    continue

                out_file.write(fen + "\n")
                kept += 1
                if progress_every and kept % progress_every == 0:
                    label = progress_label or "Kept"
                    print(f"{label}: {kept} (seen {seen})", flush=True)
                if max_positions is not None and kept >= max_positions:
                    break
    finally:
        out_file.close()

    return kept


_LICHESS_STANDARD_RE = re.compile(r"lichess_db_standard_rated_(\d{4})-(\d{2})\.pgn\.zst")


def parse_lichess_standard_index(html: str) -> list[tuple[int, int, str]]:
    """Return (year, month, filename) entries from the Lichess standard index HTML."""
    matches = []
    for match in _LICHESS_STANDARD_RE.finditer(html):
        year = int(match.group(1))
        month = int(match.group(2))
        filename = match.group(0)
        matches.append((year, month, filename))
    return matches


def latest_lichess_standard_filename(html: str) -> str | None:
    """Return the latest standard rated PGN filename from index HTML."""
    entries = parse_lichess_standard_index(html)
    if not entries:
        return None
    year, month, filename = max(entries, key=lambda t: (t[0], t[1]))
    return filename


def parse_sha256_sums(text: str) -> dict[str, str]:
    """Parse sha256sums.txt content into filename -> sha256 mapping."""
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        checksum = parts[0]
        filename = parts[-1]
        mapping[filename] = checksum
    return mapping


def parse_time_control(tc: str) -> tuple[int, int] | None:
    """Parse a Lichess-style time control string like '600+5' into (base, increment)."""
    if not tc or tc in {"-", "?"}:
        return None
    if "+" in tc:
        base_str, inc_str = tc.split("+", 1)
    else:
        base_str, inc_str = tc, "0"
    try:
        base = int(base_str)
        inc = int(inc_str)
    except ValueError:
        return None
    if base < 0 or inc < 0:
        return None
    return base, inc


def lichess_time_class(tc: str) -> str | None:
    """Return Lichess time-control class based on estimated duration."""
    parsed = parse_time_control(tc)
    if parsed is None:
        return None
    base, inc = parsed
    estimate = base + 40 * inc
    if estimate < 30:
        return "ultrabullet"
    if estimate < 180:
        return "bullet"
    if estimate < 480:
        return "blitz"
    if estimate < 1500:
        return "rapid"
    return "classical"


def _normalize_time_classes(time_class: str | Sequence[str] | None) -> set[str] | None:
    if time_class is None:
        return None
    if isinstance(time_class, str):
        parts = [part.strip() for part in time_class.split(",") if part.strip()]
        return set(parts) if parts else None
    return {str(part) for part in time_class}


def _game_passes_filters(
    headers: chess.pgn.Headers,
    *,
    min_elo: int | None,
    time_class: str | Sequence[str] | None,
    require_rated: bool | None,
    require_standard: bool,
) -> bool:
    if require_standard:
        variant = headers.get("Variant", "Standard")
        if variant.lower() != "standard":
            return False

    if require_rated is not None:
        event = headers.get("Event", "")
        is_rated = "rated" in event.lower()
        if require_rated and not is_rated:
            return False
        if not require_rated and is_rated:
            return False

    if min_elo is not None:
        try:
            w_elo = int(headers.get("WhiteElo", "0"))
            b_elo = int(headers.get("BlackElo", "0"))
        except ValueError:
            return False
        if w_elo < min_elo or b_elo < min_elo:
            return False

    time_classes = _normalize_time_classes(time_class)
    if time_classes:
        tc = headers.get("TimeControl", "")
        tc_class = lichess_time_class(tc)
        if tc_class not in time_classes:
            return False

    return True


def filter_pgn_stream(
    pgn,
    *,
    out_pgn_file=None,
    out_fens_file=None,
    max_games: int | None = None,
    ply_stride: int = 1,
    min_elo: int | None = None,
    time_class: str | Sequence[str] | None = None,
    require_rated: bool | None = None,
    require_standard: bool = True,
    progress_every: int | None = None,
    progress_label: str | None = None,
) -> int:
    """Filter PGN games from a file-like object, optionally emitting PGN/FENs."""
    if ply_stride < 1:
        raise ValueError("ply_stride must be >= 1")

    exporter = chess.pgn.FileExporter(out_pgn_file) if out_pgn_file else None
    kept = 0
    seen = 0

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        seen += 1
        headers = game.headers

        if not _game_passes_filters(
            headers,
            min_elo=min_elo,
            time_class=time_class,
            require_rated=require_rated,
            require_standard=require_standard,
        ):
            continue

        if exporter:
            game.accept(exporter)

        if out_fens_file:
            board = game.board()
            ply_idx = 0
            for move in game.mainline_moves():
                board.push(move)
                if ply_idx % ply_stride == 0:
                    out_fens_file.write(board.fen() + "\n")
                ply_idx += 1

        kept += 1
        if progress_every and kept % progress_every == 0:
            label = progress_label or "Kept"
            print(f"{label}: {kept} (seen {seen})", flush=True)
        if max_games is not None and kept >= max_games:
            break

    return kept


def filter_pgn(
    pgn_path: str,
    *,
    out_pgn: str | None = None,
    out_fens: str | None = None,
    max_games: int | None = None,
    ply_stride: int = 1,
    min_elo: int | None = None,
    time_class: str | Sequence[str] | None = None,
    require_rated: bool | None = None,
    require_standard: bool = True,
) -> int:
    """Filter PGN games by rating/time control and optionally emit FENs."""
    if ply_stride < 1:
        raise ValueError("ply_stride must be >= 1")

    out_pgn_file = open(out_pgn, "w", encoding="utf-8") if out_pgn else None
    out_fens_file = open(out_fens, "w", encoding="utf-8") if out_fens else None

    try:
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn:
            kept = filter_pgn_stream(
                pgn,
                out_pgn_file=out_pgn_file,
                out_fens_file=out_fens_file,
                max_games=max_games,
                ply_stride=ply_stride,
                min_elo=min_elo,
                time_class=time_class,
                require_rated=require_rated,
                require_standard=require_standard,
            )
    finally:
        if out_pgn_file:
            out_pgn_file.close()
        if out_fens_file:
            out_fens_file.close()

    return kept
