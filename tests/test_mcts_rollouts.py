import chess
import chess.engine

from lc0jax.interpretability.mcts_rollouts import (
    activation_records_for_line,
    board_from_root_history,
    build_rollout_pair_record,
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
        "seldepth": 9,
        "nodes": 800,
        "nps": 12345,
        "hashfull": 77,
        "tbhits": 2,
        "multipv": 1,
        "wdl": chess.engine.PovWdl(chess.engine.Wdl(501, 300, 199), chess.WHITE),
        "pv": pv,
    }
    line = line_from_info(board, info, max_depth=1)
    assert line is not None
    assert line.move == "g1f3"
    assert line.score_cp == 42
    assert line.depth == 7
    assert line.nodes == 800
    assert line.seldepth == 9
    assert line.nps == 12345
    assert line.hashfull == 77
    assert line.tbhits == 2
    assert line.multipv_rank == 1
    assert line.wdl == {"wins": 501, "draws": 300, "losses": 199}
    assert line.raw_info_keys == [
        "depth",
        "hashfull",
        "multipv",
        "nodes",
        "nps",
        "pv",
        "score",
        "seldepth",
        "tbhits",
        "wdl",
    ]
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


def test_activation_records_for_line_include_pre_root_history():
    board = chess.Board()
    pre_root = [board.fen()]
    board.push_san("e4")
    root_fen = board.fen()
    pre_root.append(root_fen)
    pv = [chess.Move.from_uci("e7e5"), chess.Move.from_uci("g1f3")]
    info = {
        "score": chess.engine.PovScore(chess.engine.Cp(42), chess.BLACK),
        "pv": pv,
    }
    line = line_from_info(board, info)

    records = activation_records_for_line(
        line,
        line_id="root_00000000:best",
        history_len=3,
        root_history_fens=pre_root,
        root_game_id="game-a",
        root_game_index=4,
        root_ply=board.ply(),
        root_source="source.pgn",
        root_record_id="record-a",
    )

    assert records[0]["fen"] == root_fen
    assert records[0]["history_fens"] == pre_root
    assert records[1]["history_fens"] == [pre_root[0], root_fen, line.fens[1]]
    assert records[2]["history_fens"] == [root_fen, line.fens[1], line.fens[2]]
    assert records[0]["root_game_id"] == "game-a"
    assert records[0]["root_game_index"] == 4
    assert records[0]["root_ply"] == board.ply()
    assert records[0]["root_source"] == "source.pgn"
    assert records[0]["root_record_id"] == "record-a"
    assert records[2]["ply"] == board.ply() + 2


def test_build_rollout_pair_record_reconstructs_engine_root_history():
    board = chess.Board()
    start_fen = board.fen()
    board.push_san("e4")
    root_fen = board.fen()
    best = chess.Move.from_uci("e7e5")
    subpar = chess.Move.from_uci("c7c5")

    reconstructed, ok = board_from_root_history(root_fen, [start_fen, root_fen])
    assert ok is True
    assert reconstructed.fen() == root_fen
    assert [move.uci() for move in reconstructed.move_stack] == ["e2e4"]

    class _Engine:
        def analyse(self, root_board, limit, *, multipv):
            assert root_board.fen() == root_fen
            assert [move.uci() for move in root_board.move_stack] == ["e2e4"]
            assert limit.nodes == 800
            assert multipv == 2
            return [
                {
                    "multipv": 1,
                    "score": chess.engine.PovScore(chess.engine.Cp(50), chess.BLACK),
                    "pv": [best],
                },
                {
                    "multipv": 2,
                    "score": chess.engine.PovScore(chess.engine.Cp(0), chess.BLACK),
                    "pv": [subpar],
                },
            ]

    record = build_rollout_pair_record(
        _Engine(),
        root_fen,
        chess.engine.Limit(nodes=800),
        multipv=2,
        min_delta_cp=1,
        root_history_fens=[start_fen, root_fen],
        search_metadata={"nodes": 800, "multipv": 2},
    )

    assert record is not None
    assert record.root_history_reconstructed is True
    assert record.search_metadata == {"nodes": 800, "multipv": 2}
    assert record.best.move == "e7e5"
    assert record.subpar[0].move == "c7c5"
    assert record.subpar[0].score_delta_cp == 50


def test_score_cp_uses_side_to_move_perspective():
    info = {"score": chess.engine.PovScore(chess.engine.Cp(25), chess.WHITE)}
    assert score_cp_from_info(info, turn=chess.WHITE) == 25
    assert score_cp_from_info(info, turn=chess.BLACK) == -25
