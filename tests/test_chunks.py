import numpy as np
import chess

from lc0jax.training.chunks import TrainingRecord, record_to_board, record_to_move
from lc0jax.modeling import policy as policy_mod


def test_record_to_board_basic():
    planes = np.zeros(104, dtype=np.uint64)
    planes[3] = (1 << chess.A1) | (1 << chess.H1)  # white rooks
    planes[5] = 1 << chess.E1  # white king
    planes[9] = (1 << chess.A8) | (1 << chess.H8)  # black rooks
    planes[11] = 1 << chess.E8  # black king
    record = TrainingRecord(
        version=6,
        input_format=3,
        planes=planes,
        castling=(1, 1, 1, 1),
        side_to_move=0,
        rule50=0,
        invariance_info=0,
        played_idx=None,
        best_idx=None,
    )

    board = record_to_board(record)
    assert board.piece_at(chess.E1).symbol() == "K"
    assert board.piece_at(chess.E8).symbol() == "k"
    assert board.has_kingside_castling_rights(chess.WHITE)
    assert board.has_queenside_castling_rights(chess.WHITE)
    assert board.has_kingside_castling_rights(chess.BLACK)
    assert board.has_queenside_castling_rights(chess.BLACK)


def test_record_to_move_identity():
    planes = np.zeros(104, dtype=np.uint64)
    move_idx = policy_mod.move_to_policy_index("e2e4", "lc0_1858")
    record = TrainingRecord(
        version=6,
        input_format=3,
        planes=planes,
        castling=(0, 0, 0, 0),
        side_to_move=0,
        rule50=0,
        invariance_info=0,
        played_idx=move_idx,
        best_idx=None,
    )
    move = record_to_move(record)
    assert move == chess.Move.from_uci("e2e4")
