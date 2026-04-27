import numpy as np

from lc0jax.interpretability.dynamic_baselines import (
    dynamic_baseline_report,
    evaluate_direction,
    random_sparse_directions,
)


def test_evaluate_direction_reports_constraint_satisfaction():
    differences = np.asarray([[2.0, 0.0], [-1.0, 0.0]])
    direction = np.asarray([1.0, 0.0])
    metrics = evaluate_direction(differences, direction, margin=1.0)

    assert metrics["count"] == 2
    assert metrics["constraint_satisfaction"] == 0.5
    assert metrics["margin_satisfaction"] == 0.5
    assert metrics["mean_score"] == 0.5


def test_random_sparse_directions_are_unit_norm_and_sparse():
    directions = random_sparse_directions(count=4, dimension=8, nonzero=2, seed=0)
    assert directions.shape == (4, 8)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.all(np.count_nonzero(directions, axis=1) == 2)


def test_dynamic_baseline_report_contains_random_and_shuffled_summaries():
    differences = np.tile(np.asarray([2.0, 0.0, 0.0]), (4, 1))
    direction = np.asarray([1.0, 0.0, 0.0])

    report = dynamic_baseline_report(
        differences,
        direction,
        random_count=3,
        shuffled_label_count=5,
        seed=0,
    )

    assert report["actual"]["constraint_satisfaction"] == 1.0
    assert report["nonzero_features"] == 1
    assert report["random_sparse"]["count"] == 3
    assert report["shuffled_labels"]["count"] == 5
    assert report["shuffled_solve"]["count"] == 0
