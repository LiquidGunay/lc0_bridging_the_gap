"""Parse LC0 training chunks (v3/v6) and convert to boards/moves."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
from typing import Iterator

import numpy as np

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None

from lc0jax.modeling import policy as policy_mod
from lc0jax.modeling import encode as encode_mod


V3_RECORD_SIZE = 8276
V4_RECORD_SIZE = 8292
V5_RECORD_SIZE = 8308
V6_RECORD_SIZE = 8356


@dataclass
class TrainingRecord:
    version: int
    input_format: int
    planes: np.ndarray
    castling: tuple[int, int, int, int]
    side_to_move: int
    rule50: int
    invariance_info: int
    played_idx: int | None
    best_idx: int | None


def _open_chunk(path: str) -> io.BufferedReader:
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if path.endswith(".zst"):
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("zstandard is required to read .zst chunks.") from exc
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return dctx.stream_reader(fh)
    return open(path, "rb")


def iter_records(path: str) -> Iterator[TrainingRecord]:
    with _open_chunk(path) as f:
        while True:
            version_bytes = f.read(4)
            if len(version_bytes) < 4:
                break
            version = int.from_bytes(version_bytes, "little", signed=False)
            if version == 3:
                record_size = V3_RECORD_SIZE
            elif version == 4:
                record_size = V4_RECORD_SIZE
            elif version == 5:
                record_size = V5_RECORD_SIZE
            elif version == 6:
                record_size = V6_RECORD_SIZE
            else:
                raise ValueError(f"Unsupported chunk version: {version}")
            rest = f.read(record_size - 4)
            if len(rest) < record_size - 4:
                break
            record = version_bytes + rest

            offset = 4
            input_format = 0
            if version >= 5:
                input_format = int.from_bytes(record[offset : offset + 4], "little")
                offset += 4

            probs_offset = offset
            planes_offset = probs_offset + 1858 * 4
            planes = np.frombuffer(record, dtype="<u8", count=104, offset=planes_offset).copy()

            castling_offset = planes_offset + 104 * 8
            castling = tuple(int(b) for b in record[castling_offset : castling_offset + 4])

            stm_offset = castling_offset + 4
            side_to_move = int(record[stm_offset])
            rule50 = int(record[stm_offset + 1])
            invariance_info = int(record[stm_offset + 2]) if version >= 5 else 0

            played_idx = None
            best_idx = None
            if version >= 6:
                floats_offset = stm_offset + 4
                visits_offset = floats_offset + 15 * 4
                played_idx = int.from_bytes(record[visits_offset + 4 : visits_offset + 6], "little")
                best_idx = int.from_bytes(record[visits_offset + 6 : visits_offset + 8], "little")

                # For input type 3, side-to-move is encoded in invariance_info bit 7.
                if input_format == 3:
                    side_to_move = (invariance_info >> 7) & 1
            elif version >= 5 and input_format == 3:
                side_to_move = (invariance_info >> 7) & 1

            yield TrainingRecord(
                version=version,
                input_format=input_format,
                planes=planes,
                castling=castling,
                side_to_move=side_to_move,
                rule50=rule50,
                invariance_info=invariance_info,
                played_idx=played_idx,
                best_idx=best_idx,
            )


def _bitboards_from_planes(planes: np.ndarray) -> tuple[list[int], list[int]]:
    ours = [int(x) for x in planes[0:6]]
    theirs = [int(x) for x in planes[6:12]]
    return ours, theirs


def _apply_transform(mask: int, transform: int) -> int:
    return encode_mod._transform_mask(mask, transform)


def _reverse_bytes(mask: int) -> int:
    return encode_mod._reverse_bytes_in_bytes(mask)


def record_to_board(record: TrainingRecord) -> "chess.Board":
    if chess is None:
        raise ImportError("python-chess is required for chunk conversion.")

    planes = record.planes.copy()
    transform = record.invariance_info & 0x07
    if transform:
        for idx in range(12):
            planes[idx] = _apply_transform(int(planes[idx]), transform)

    ours, theirs = _bitboards_from_planes(planes)
    if record.side_to_move == 1:
        ours = [_reverse_bytes(bb) for bb in ours]
        theirs = [_reverse_bytes(bb) for bb in theirs]

    board = chess.Board(None)
    if record.side_to_move == 0:
        white_planes = ours
        black_planes = theirs
    else:
        white_planes = theirs
        black_planes = ours

    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for plane, ptype in zip(white_planes, piece_types):
        bb = plane
        while bb:
            lsb = bb & -bb
            sq = (lsb.bit_length() - 1)
            board.set_piece_at(sq, chess.Piece(ptype, chess.WHITE))
            bb &= bb - 1

    for plane, ptype in zip(black_planes, piece_types):
        bb = plane
        while bb:
            lsb = bb & -bb
            sq = (lsb.bit_length() - 1)
            board.set_piece_at(sq, chess.Piece(ptype, chess.BLACK))
            bb &= bb - 1

    board.turn = chess.WHITE if record.side_to_move == 0 else chess.BLACK

    us_ooo, us_oo, them_ooo, them_oo = record.castling
    rights = 0
    if record.side_to_move == 0:
        if us_ooo:
            rights |= chess.BB_A1
        if us_oo:
            rights |= chess.BB_H1
        if them_ooo:
            rights |= chess.BB_A8
        if them_oo:
            rights |= chess.BB_H8
    else:
        if us_ooo:
            rights |= chess.BB_A8
        if us_oo:
            rights |= chess.BB_H8
        if them_ooo:
            rights |= chess.BB_A1
        if them_oo:
            rights |= chess.BB_H1
    board.castling_rights = rights
    board.halfmove_clock = int(record.rule50)
    board.fullmove_number = 1
    return board


def _transform_square(square: int, transform: int, side_to_move: int) -> int:
    mask = 1 << square
    if transform:
        mask = _apply_transform(mask, transform)
    if side_to_move == 1:
        mask = _reverse_bytes(mask)
    return mask.bit_length() - 1


def record_to_move(record: TrainingRecord, *, use_best: bool = False) -> "chess.Move" | None:
    if chess is None:
        raise ImportError("python-chess is required for chunk conversion.")
    idx = record.best_idx if use_best else record.played_idx
    if idx is None:
        return None
    try:
        move = policy_mod.policy_index_to_move(idx, "lc0_1858")
    except Exception:
        return None

    transform = record.invariance_info & 0x07
    from_sq = _transform_square(move.from_square, transform, record.side_to_move)
    to_sq = _transform_square(move.to_square, transform, record.side_to_move)
    return chess.Move(from_sq, to_sq, promotion=move.promotion)
