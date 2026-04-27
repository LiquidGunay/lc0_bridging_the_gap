import numpy as np

from lc0jax.interpretability.novelty import novelty_curve, reconstruction_loss, right_svd_basis


def test_reconstruction_loss_is_low_for_aligned_basis():
    basis = np.eye(3)
    assert reconstruction_loss(np.array([1.0, 0.0, 0.0]), basis, rank=1) == 0.0


def test_novelty_curve_prefers_machine_basis_when_aligned():
    machine = np.zeros((8, 3), dtype=np.float64)
    human = np.zeros((8, 3), dtype=np.float64)
    machine[:, 0] = np.linspace(-1.0, 1.0, 8)
    human[:, 1] = np.linspace(-1.0, 1.0, 8)
    report = novelty_curve(np.array([1.0, 0.0, 0.0]), machine, human, ranks=[1])
    assert report[0]["novelty_area"] > 0
    assert report[0]["positive_rank_fraction"] == 1.0


def test_right_svd_basis_flattens_token_activations():
    tokens = np.zeros((4, 64, 2), dtype=np.float64)
    tokens[:, :, 0] = np.arange(4)[:, None]
    basis = right_svd_basis(tokens, max_rank=1)
    assert basis.shape == (1, 128)
