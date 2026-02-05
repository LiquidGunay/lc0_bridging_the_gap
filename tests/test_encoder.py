import numpy as np
import chess

from lc0jax.encode import encode_board, AUX_PLANE_BASE


def test_encode_deterministic():
    board = chess.Board()
    planes1 = encode_board(board, history=[], planes_layout="nchw", input_format="INPUT_CLASSICAL_112_PLANE")
    planes2 = encode_board(board, history=[], planes_layout="nchw", input_format="INPUT_CLASSICAL_112_PLANE")
    assert np.array_equal(planes1, planes2)


def test_castling_planes_change():
    board = chess.Board()
    no_castling = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")

    planes_castle = encode_board(board, history=[], planes_layout="nchw", input_format="INPUT_112_WITH_CANONICALIZATION_V2")
    planes_no_castle = encode_board(no_castling, history=[], planes_layout="nchw", input_format="INPUT_112_WITH_CANONICALIZATION_V2")

    assert not np.array_equal(planes_castle[AUX_PLANE_BASE + 0], planes_no_castle[AUX_PLANE_BASE + 0])
    assert not np.array_equal(planes_castle[AUX_PLANE_BASE + 1], planes_no_castle[AUX_PLANE_BASE + 1])


def test_en_passant_plane():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1")
    planes = encode_board(board, history=[], planes_layout="nchw", input_format="INPUT_112_WITH_CANONICALIZATION_V2")
    ep_plane = planes[AUX_PLANE_BASE + 4]
    assert ep_plane.sum() == 1.0


def test_side_to_move_plane_classical():
    black_to_move = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    planes = encode_board(black_to_move, history=[], planes_layout="nchw", input_format="INPUT_CLASSICAL_112_PLANE")
    stm_plane = planes[AUX_PLANE_BASE + 4]
    assert stm_plane.sum() == 64.0
