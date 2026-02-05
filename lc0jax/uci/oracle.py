"""ONNX oracle runner."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort


def run_onnx(onnx_path: str, planes: np.ndarray) -> dict:
    """Return ONNX outputs as numpy arrays with stable keys."""
    if planes.ndim != 4:
        raise ValueError(f"Expected planes with rank 4, got shape {planes.shape}")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_info = sess.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    data = planes
    if len(input_shape) == 4:
        # Infer layout: if channel dim is last, convert NCHW -> NHWC.
        # If channel dim is second, leave as is.
        c_dim = input_shape.index("C") if "C" in input_shape else None
        if c_dim is None:
            # Heuristic: if expected last dim matches our channel count, treat as NHWC.
            if input_shape[-1] == planes.shape[1]:
                data = np.transpose(planes, (0, 2, 3, 1))
        else:
            if c_dim == 3:
                data = np.transpose(planes, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unexpected ONNX input rank: {input_shape}")

    if input_info.type == "tensor(float16)":
        data = data.astype(np.float16, copy=False)
    else:
        data = data.astype(np.float32, copy=False)

    outputs = sess.run(None, {input_name: data})
    output_names = [out.name for out in sess.get_outputs()]
    return {name: value for name, value in zip(output_names, outputs)}
