import jax.numpy as jnp
import numpy as np

from lc0jax.modeling.model import _reshape_patch_vector


def test_reshape_patch_vector_keeps_channel_vector_broadcastable():
    value = jnp.zeros((128, 3), dtype=jnp.float32)
    patch = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    shaped = _reshape_patch_vector(value, patch)

    assert shaped.shape == (3,)
    np.testing.assert_array_equal(np.asarray(shaped), [1.0, 2.0, 3.0])


def test_reshape_patch_vector_repeats_flat_square_local_vector_per_batch():
    value = jnp.zeros((128, 3), dtype=jnp.float32)
    patch = jnp.arange(64 * 3, dtype=jnp.float32)

    shaped = _reshape_patch_vector(value, patch)

    assert shaped.shape == (128, 3)
    np.testing.assert_array_equal(np.asarray(shaped[:64]), np.arange(64 * 3).reshape(64, 3))
    np.testing.assert_array_equal(np.asarray(shaped[64:]), np.arange(64 * 3).reshape(64, 3))
