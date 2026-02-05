import chess

from lc0jax.policy import legal_move_mask, move_to_policy_index


def test_legal_move_mask_marks_legal():
    board = chess.Board()
    mask = legal_move_mask(board, policy_format="lc0")
    for move in board.legal_moves:
        idx = move_to_policy_index(move, policy_format="lc0")
        assert mask[idx]


def test_move_to_index_roundtrip():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = move_to_policy_index(move, policy_format="lc0")
    assert isinstance(idx, int)
