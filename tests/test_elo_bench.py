from lc0jax.uci.elo import elo_from_score


def test_elo_from_score_symmetry():
    assert abs(elo_from_score(0.5)) < 1e-6
    assert elo_from_score(0.75) > 0
    assert elo_from_score(0.25) < 0
