import chess
import chess.engine

from lc0jax.interpretability.mcts_rollouts import line_from_info, pv_to_fens, score_cp_from_info


def test_pv_to_fens_replays_root_plus_moves():
    board = chess.Board()
    pv = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
    fens = pv_to_fens(board, pv)
    assert len(fens) == 3
    assert fens[0] == chess.Board().fen()
    replayed = chess.Board()
    replayed.push(pv[0])
    replayed.push(pv[1])
    assert fens[-1] == replayed.fen()


def test_line_from_info_serializes_score_and_pv():
    board = chess.Board()
    pv = [chess.Move.from_uci("g1f3"), chess.Move.from_uci("g8f6")]
    info = {
        "score": chess.engine.PovScore(chess.engine.Cp(42), chess.WHITE),
        "depth": 7,
        "nodes": 800,
        "pv": pv,
    }
    line = line_from_info(board, info, max_depth=1)
    assert line is not None
    assert line.move == "g1f3"
    assert line.score_cp == 42
    assert line.depth == 7
    assert line.nodes == 800
    assert line.pv == ["g1f3"]
    assert len(line.fens) == 2


def test_score_cp_uses_side_to_move_perspective():
    info = {"score": chess.engine.PovScore(chess.engine.Cp(25), chess.WHITE)}
    assert score_cp_from_info(info, turn=chess.WHITE) == 25
    assert score_cp_from_info(info, turn=chess.BLACK) == -25
