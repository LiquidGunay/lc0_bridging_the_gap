import numpy as np

from lc0jax.interpretability.dynamic_causal import policy_margin_report


def test_policy_margin_report_summarizes_patch_effects():
    base_policy = np.asarray(
        [
            [0.0, 2.0, 1.0],
            [3.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    patched_policy = np.asarray(
        [
            [0.0, 3.0, 1.0],
            [2.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )

    report = policy_margin_report(
        base_policy=base_policy,
        patched_policy=patched_policy,
        best_indices=[1, 2],
        subpar_indices=[2, 0],
        root_fens=["fen-a", "fen-b"],
        best_moves=["a", "b"],
        subpar_moves=["c", "d"],
    )

    assert report["num_pairs"] == 2
    assert report["mean_base_margin"] == -0.5
    assert report["mean_patched_margin"] == 2.0
    assert report["mean_delta_margin"] == 2.5
    assert report["fraction_delta_positive"] == 1.0
    assert report["top1_change_rate"] == 0.5
    assert report["examples"][0]["delta_margin"] == 1.0
