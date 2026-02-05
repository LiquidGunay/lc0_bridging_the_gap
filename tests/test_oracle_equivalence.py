import os
import numpy as np
import pytest

from lc0jax.encode import encode_board
from lc0jax.oracle import run_onnx


ONNX_PATH = "models/BT4.onnx"
FENS_PATH = "data/fens.txt"


@pytest.mark.skipif(not os.path.exists(ONNX_PATH), reason="ONNX model not available")
def test_oracle_shapes_and_nans():
    with open(FENS_PATH, "r", encoding="utf-8") as f:
        fens = [line.strip() for line in f if line.strip()]
    planes = np.stack([encode_board(fen, history=[]) for fen in fens], axis=0)
    outputs = run_onnx(ONNX_PATH, planes)

    assert "/output/policy" in outputs
    assert "/output/wdl" in outputs
    assert outputs["/output/policy"].shape[0] == len(fens)
    assert outputs["/output/policy"].shape[1] == 1858
    assert outputs["/output/wdl"].shape[1] == 3
    assert not np.isnan(outputs["/output/policy"]).any()
    assert not np.isnan(outputs["/output/wdl"]).any()
