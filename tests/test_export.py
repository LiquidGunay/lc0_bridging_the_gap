import numpy as np

from lc0jax.uci import export as export_mod
from lc0jax.modeling.weights import _decode_layer, _reshape_mat
from lc0jax.proto import net_pb2


def test_linear16_roundtrip_vec():
    layer = net_pb2.Weights.Layer()
    vec = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    export_mod._set_layer(layer, vec, kind="vec", encoding="LINEAR16")
    decoded = _decode_layer(layer, layer.LINEAR16)
    assert decoded.shape == vec.shape
    assert np.allclose(decoded, vec, atol=1e-2)


def test_linear16_roundtrip_mat():
    layer = net_pb2.Weights.Layer()
    mat = np.arange(12, dtype=np.float32).reshape(3, 4)
    export_mod._set_layer(layer, mat, kind="mat", encoding="LINEAR16")
    decoded = _decode_layer(layer, layer.LINEAR16)
    restored = _reshape_mat(decoded, rows=mat.shape[0], cols=mat.shape[1])
    assert restored.shape == mat.shape
    assert np.allclose(restored, mat, atol=1e-2)
