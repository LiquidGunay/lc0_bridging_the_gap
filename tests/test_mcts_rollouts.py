import chess
import chess.engine

from lc0jax.interpretability.mcts_rollouts import (
    activation_records_for_line,
    line_from_info,
    pv_to_fens,
    score_cp_from_info,
)


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
    assert line.activation_keys is None


def test_activation_records_for_line_preserve_rolling_history_and_keys():
    board = chess.Board()
    pv = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
    ]
    info = {
        "score": chess.engine.PovScore(chess.engine.Cp(42), chess.WHITE),
        "pv": pv,
    }
    line = line_from_info(board, info)
    records = activation_records_for_line(line, line_id="root_00000000:best", history_len=2)

    assert line.activation_keys == [
        "root_00000000:best:000",
        "root_00000000:best:001",
        "root_00000000:best:002",
        "root_00000000:best:003",
    ]
    assert [record["activation_key"] for record in records] == line.activation_keys
    assert records[0]["history_fens"] == [line.fens[0]]
    assert records[1]["history_fens"] == line.fens[0:2]
    assert records[2]["history_fens"] == line.fens[1:3]
    assert records[3]["history_fens"] == line.fens[2:4]


def test_score_cp_uses_side_to_move_perspective():
    info = {"score": chess.engine.PovScore(chess.engine.Cp(25), chess.WHITE)}
    assert score_cp_from_info(info, turn=chess.WHITE) == 25
    assert score_cp_from_info(info, turn=chess.BLACK) == -25
