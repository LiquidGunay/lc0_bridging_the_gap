from lc0jax.interpretability import datasets


def test_latest_lichess_standard_filename():
    html = """
    <a href="lichess_db_standard_rated_2024-12.pgn.zst">lichess_db_standard_rated_2024-12.pgn.zst</a>
    <a href="lichess_db_standard_rated_2025-01.pgn.zst">lichess_db_standard_rated_2025-01.pgn.zst</a>
    <a href="lichess_db_standard_rated_2025-12.pgn.zst">lichess_db_standard_rated_2025-12.pgn.zst</a>
    """
    assert datasets.latest_lichess_standard_filename(html) == "lichess_db_standard_rated_2025-12.pgn.zst"


def test_parse_sha256_sums():
    text = """
    abc123  lichess_db_standard_rated_2025-12.pgn.zst
    def456  lichess_db_standard_rated_2025-11.pgn.zst
    """
    mapping = datasets.parse_sha256_sums(text)
    assert mapping["lichess_db_standard_rated_2025-12.pgn.zst"] == "abc123"
    assert mapping["lichess_db_standard_rated_2025-11.pgn.zst"] == "def456"


def test_lichess_time_class():
    assert datasets.lichess_time_class("60+1") == "bullet"
    assert datasets.lichess_time_class("300+0") == "blitz"
    assert datasets.lichess_time_class("600+5") == "rapid"
    assert datasets.lichess_time_class("1800+0") == "classical"


def test_filter_pgn_stream():
    import io

    pgn_text = """[Event \"Rated Classical game\"]\n[Site \"https://lichess.org/abc\"]\n[Date \"2025.12.01\"]\n[Round \"-\"]\n[White \"A\"]\n[Black \"B\"]\n[Result \"1-0\"]\n[WhiteElo \"2500\"]\n[BlackElo \"2500\"]\n[TimeControl \"1800+0\"]\n[Variant \"Standard\"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n\n[Event \"Rated Blitz game\"]\n[Site \"https://lichess.org/def\"]\n[Date \"2025.12.01\"]\n[Round \"-\"]\n[White \"C\"]\n[Black \"D\"]\n[Result \"1/2-1/2\"]\n[WhiteElo \"2600\"]\n[BlackElo \"2600\"]\n[TimeControl \"300+0\"]\n[Variant \"Standard\"]\n\n1. d4 d5 1/2-1/2\n"""
    pgn_stream = io.StringIO(pgn_text)
    out_pgn = io.StringIO()

    kept = datasets.filter_pgn_stream(
        pgn_stream,
        out_pgn_file=out_pgn,
        max_games=10,
        min_elo=2400,
        time_class="classical",
        require_rated=True,
        require_standard=True,
    )

    assert kept == 1
    assert "Rated Classical game" in out_pgn.getvalue()


def test_filter_pgn_stream_multi_time_class():
    import io

    pgn_text = """[Event \"Rated Rapid game\"]\n[Site \"https://lichess.org/ghi\"]\n[Date \"2025.12.01\"]\n[Round \"-\"]\n[White \"E\"]\n[Black \"F\"]\n[Result \"1-0\"]\n[WhiteElo \"2450\"]\n[BlackElo \"2450\"]\n[TimeControl \"600+5\"]\n[Variant \"Standard\"]\n\n1. e4 e5 1-0\n\n[Event \"Rated Classical game\"]\n[Site \"https://lichess.org/jkl\"]\n[Date \"2025.12.01\"]\n[Round \"-\"]\n[White \"G\"]\n[Black \"H\"]\n[Result \"1-0\"]\n[WhiteElo \"2500\"]\n[BlackElo \"2500\"]\n[TimeControl \"1800+0\"]\n[Variant \"Standard\"]\n\n1. d4 d5 1-0\n"""
    pgn_stream = io.StringIO(pgn_text)
    out_pgn = io.StringIO()

    kept = datasets.filter_pgn_stream(
        pgn_stream,
        out_pgn_file=out_pgn,
        max_games=10,
        min_elo=2400,
        time_class=["rapid", "classical"],
        require_rated=True,
        require_standard=True,
    )

    assert kept == 2


def test_filter_fens_min_ply(tmp_path):
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ]
    in_path = tmp_path / "in.fens"
    out_path = tmp_path / "out.fens"
    in_path.write_text("\n".join(fens) + "\n", encoding="utf-8")

    kept = datasets.filter_fens(str(in_path), out_fens=str(out_path), min_ply=2)
    assert kept == 1
    out_lines = out_path.read_text(encoding="utf-8").splitlines()
    assert out_lines == [fens[1]]


def test_filter_fens_phase(tmp_path):
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/8/8/8/8/4k3/8/4K3 w - - 0 1",
    ]
    in_path = tmp_path / "in.fens"
    out_path = tmp_path / "out.fens"
    in_path.write_text("\n".join(fens) + "\n", encoding="utf-8")

    kept = datasets.filter_fens(str(in_path), out_fens=str(out_path), max_phase=0.2)
    assert kept == 1
    out_lines = out_path.read_text(encoding="utf-8").splitlines()
    assert out_lines == [fens[1]]
