import numpy as np

from lc0jax.interpretability.concepts import discover_concepts


def test_cov_shift_shapes():
    rng = np.random.default_rng(0)
    emb_a = rng.standard_normal((32, 16))
    emb_b = rng.standard_normal((32, 16))
    result = discover_concepts(emb_a, emb_b, method="cov_shift", k=3)
    direction = result["direction"]
    assert direction.shape == (16, 3)
    scores = result["scores"]
    assert scores is not None
    assert len(scores) == 3


def test_cluster_diff_shapes():
    rng = np.random.default_rng(1)
    emb_a = rng.standard_normal((40, 12))
    emb_b = rng.standard_normal((40, 12))
    result = discover_concepts(emb_a, emb_b, method="cluster_diff", k=4)
    direction = result["direction"]
    assert direction.shape == (12, 4)
    scores = result["scores"]
    assert scores is not None
    assert len(scores) == 4
