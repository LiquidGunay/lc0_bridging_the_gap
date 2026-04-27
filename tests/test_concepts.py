import numpy as np

from lc0jax.interpretability.concepts import (
    discover_concepts,
    dynamic_rollout_differences,
    screen_sparse_concept_features,
    solve_screened_sparse_concept_from_differences,
    solve_sparse_concept_from_differences,
)


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


def test_sparse_concept_from_differences_recovers_positive_margin():
    differences = np.tile(np.array([2.0, 0.0, 0.0]), (8, 1))
    result = solve_sparse_concept_from_differences(differences, standardize=False)
    assert result["direction"][0] > 0.99
    assert result["constraint_satisfaction"] == 1.0
    assert result["margin_satisfaction"] == 1.0


def test_feature_screening_selects_high_abs_mean_features():
    differences = np.asarray([
        [0.0, 3.0, 0.1, -2.0],
        [0.0, 3.0, -0.1, -2.0],
        [0.0, 3.0, 0.0, -2.0],
    ])

    selected, scores = screen_sparse_concept_features(
        differences,
        max_features=2,
        standardize=False,
    )

    np.testing.assert_array_equal(selected, np.asarray([1, 3]))
    assert scores[1] == 3.0
    assert scores[3] == 2.0


def test_screened_sparse_concept_expands_to_original_dimension():
    differences = np.tile(np.asarray([0.0, 2.0, 0.0, 0.0]), (8, 1))

    result = solve_screened_sparse_concept_from_differences(
        differences,
        max_features=1,
        standardize=False,
    )

    assert result["method"] == "screened_sparse_cvxpy"
    assert result["direction"].shape == (4,)
    assert result["screened_dimension"] == 1
    np.testing.assert_array_equal(result["screening_indices"], np.asarray([1]))
    assert result["direction"][1] > 0.99
    assert np.count_nonzero(np.abs(result["direction"]) > 1e-8) == 1
    assert result["constraint_satisfaction"] == 1.0
    assert result["margin_satisfaction"] == 1.0


def test_dynamic_rollout_differences_flat_and_single_even():
    optimal = np.zeros((2, 3, 64, 2), dtype=np.float32)
    subpar = np.zeros((2, 1, 3, 64, 2), dtype=np.float32)
    optimal[:, 0, :, 0] = 2.0
    optimal[:, 1, :, 0] = 100.0
    differences = dynamic_rollout_differences(
        optimal,
        subpar,
        mode="mean",
        index_mode="single_even",
    )
    assert differences.shape == (2, 2)
    np.testing.assert_allclose(differences[:, 0], 1.0)
    np.testing.assert_allclose(differences[:, 1], 0.0)
