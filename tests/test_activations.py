import numpy as np

from lc0jax.interpretability.activations import project_token_activations, reshape_token_activations


def test_reshape_token_activations_from_flat_capture():
    act = np.arange(2 * 64 * 3, dtype=np.float32).reshape(2 * 64, 3)
    tokens = reshape_token_activations(act, batch=2)
    assert tokens.shape == (2, 64, 3)
    np.testing.assert_array_equal(tokens[0, 0], act[0])
    np.testing.assert_array_equal(tokens[1, 0], act[64])


def test_project_token_activations_mean_and_flat():
    tokens = np.arange(2 * 64 * 3, dtype=np.float32).reshape(2, 64, 3)
    mean = project_token_activations(tokens, mode="mean")
    flat = project_token_activations(tokens, mode="flat")
    assert mean.shape == (2, 3)
    assert flat.shape == (2, 64 * 3)
    np.testing.assert_allclose(mean[0], tokens[0].mean(axis=0))
    np.testing.assert_array_equal(flat[1], tokens[1].reshape(-1))
