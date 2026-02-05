"""Export JAX BT4 parameters to LC0 .pb.gz weights."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from typing import Iterable

import numpy as np


try:
    from lc0jax.proto import net_pb2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing protobuf bindings. Run `python tools/gen_proto.py` after installing protoc."
    ) from exc


@dataclass(frozen=True)
class ExportStats:
    layers_written: int
    encoding: str


def _flatten_mat(mat: np.ndarray) -> np.ndarray:
    """Invert LC0's matmul reshape convention (rows, cols) -> flat."""
    mat = np.asarray(mat, dtype=np.float32)
    return mat.T.reshape(-1)


def _flatten_vec(vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def _encode_linear16(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    vals = np.asarray(values, dtype=np.float32)
    min_val = float(np.min(vals)) if vals.size else 0.0
    max_val = float(np.max(vals)) if vals.size else 0.0
    if max_val == min_val:
        q = np.zeros_like(vals, dtype=np.uint16)
    else:
        scale = 65535.0 / (max_val - min_val)
        q = np.round((vals - min_val) * scale).clip(0, 65535).astype(np.uint16)
    return q, min_val, max_val


def _set_layer(
    layer,
    array: np.ndarray,
    *,
    kind: str,
    encoding: str,
) -> None:
    if kind == "mat":
        flat = _flatten_mat(array)
        default_dims = [int(np.asarray(array).shape[1]), int(np.asarray(array).shape[0])]
    elif kind == "vec":
        flat = _flatten_vec(array)
        default_dims = [int(flat.size)]
    else:
        raise ValueError(f"Unsupported layer kind: {kind}")

    if layer.dims:
        expected = int(np.prod(layer.dims))
        if expected != flat.size:
            layer.dims[:] = default_dims
    else:
        layer.dims[:] = default_dims

    if encoding == "LINEAR16":
        q, min_val, max_val = _encode_linear16(flat)
        layer.min_val = min_val
        layer.max_val = max_val
        layer.params = q.tobytes()
        layer.encoding = net_pb2.Weights.Layer.LINEAR16
    elif encoding == "FLOAT16":
        layer.params = flat.astype(np.float16).tobytes()
        layer.encoding = net_pb2.Weights.Layer.FLOAT16
        if layer.HasField("min_val"):
            layer.ClearField("min_val")
        if layer.HasField("max_val"):
            layer.ClearField("max_val")
    elif encoding == "BFLOAT16":
        arr = flat.astype(np.float32)
        raw = (arr.view(np.uint32) >> 16).astype(np.uint16)
        layer.params = raw.tobytes()
        layer.encoding = net_pb2.Weights.Layer.BFLOAT16
        if layer.HasField("min_val"):
            layer.ClearField("min_val")
        if layer.HasField("max_val"):
            layer.ClearField("max_val")
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


def export_bt4_params(
    params: dict,
    *,
    template_path: str,
    out_path: str,
    encoding: str = "LINEAR16",
) -> ExportStats:
    """Export BT4 params into LC0 .pb.gz using a template for metadata."""
    net = net_pb2.Net()
    with gzip.open(template_path, "rb") as f:
        net.ParseFromString(f.read())

    net.ClearField("onnx_model")
    if not net.HasField("weights"):
        net.weights.CopyFrom(net_pb2.Weights())

    if not net.HasField("format"):
        net.format.CopyFrom(net_pb2.Format())
    net.format.weights_encoding = net_pb2.Format.LINEAR16

    w = net.weights
    w.headcount = int(params.get("headcount", w.headcount or 0))

    emb = params["embedding"]
    _set_layer(w.ip_emb_preproc_w, emb["preproc_w"], kind="mat", encoding=encoding)
    _set_layer(w.ip_emb_preproc_b, emb["preproc_b"], kind="vec", encoding=encoding)
    _set_layer(w.ip_emb_w, emb["w"], kind="mat", encoding=encoding)
    _set_layer(w.ip_emb_b, emb["b"], kind="vec", encoding=encoding)
    _set_layer(w.ip_emb_ln_gammas, emb["ln_scale"], kind="vec", encoding=encoding)
    _set_layer(w.ip_emb_ln_betas, emb["ln_bias"], kind="vec", encoding=encoding)
    _set_layer(w.ip_mult_gate, emb["mul_gate"], kind="mat", encoding=encoding)
    _set_layer(w.ip_add_gate, emb["add_gate"], kind="mat", encoding=encoding)
    _set_layer(w.ip_emb_ffn.dense1_w, emb["ffn"]["dense1_w"], kind="mat", encoding=encoding)
    _set_layer(w.ip_emb_ffn.dense1_b, emb["ffn"]["dense1_b"], kind="vec", encoding=encoding)
    _set_layer(w.ip_emb_ffn.dense2_w, emb["ffn"]["dense2_w"], kind="mat", encoding=encoding)
    _set_layer(w.ip_emb_ffn.dense2_b, emb["ffn"]["dense2_b"], kind="vec", encoding=encoding)
    _set_layer(w.ip_emb_ffn_ln_gammas, emb["ffn_ln_scale"], kind="vec", encoding=encoding)
    _set_layer(w.ip_emb_ffn_ln_betas, emb["ffn_ln_bias"], kind="vec", encoding=encoding)

    enc_layers = params["encoder"]
    if len(w.encoder) != len(enc_layers):
        if len(w.encoder) == 0:
            for _ in range(len(enc_layers)):
                w.encoder.add()
        elif len(w.encoder) != len(enc_layers):
            raise ValueError(
                f"Encoder layer mismatch: template has {len(w.encoder)} vs params {len(enc_layers)}"
            )

    for idx, layer in enumerate(enc_layers):
        enc = w.encoder[idx]
        mha = layer["mha"]
        _set_layer(enc.mha.q_w, mha["q_w"], kind="mat", encoding=encoding)
        _set_layer(enc.mha.q_b, mha["q_b"], kind="vec", encoding=encoding)
        _set_layer(enc.mha.k_w, mha["k_w"], kind="mat", encoding=encoding)
        _set_layer(enc.mha.k_b, mha["k_b"], kind="vec", encoding=encoding)
        _set_layer(enc.mha.v_w, mha["v_w"], kind="mat", encoding=encoding)
        _set_layer(enc.mha.v_b, mha["v_b"], kind="vec", encoding=encoding)
        _set_layer(enc.mha.dense_w, mha["dense_w"], kind="mat", encoding=encoding)
        _set_layer(enc.mha.dense_b, mha["dense_b"], kind="vec", encoding=encoding)
        smol = mha["smolgen"]
        _set_layer(enc.mha.smolgen.compress, smol["compress_w"], kind="mat", encoding=encoding)
        _set_layer(enc.mha.smolgen.dense1_w, smol["dense1_w"], kind="mat", encoding=encoding)
        _set_layer(enc.mha.smolgen.dense1_b, smol["dense1_b"], kind="vec", encoding=encoding)
        _set_layer(enc.mha.smolgen.ln1_gammas, smol["ln1_scale"], kind="vec", encoding=encoding)
        _set_layer(enc.mha.smolgen.ln1_betas, smol["ln1_bias"], kind="vec", encoding=encoding)
        _set_layer(enc.mha.smolgen.dense2_w, smol["dense2_w"], kind="mat", encoding=encoding)
        _set_layer(enc.mha.smolgen.dense2_b, smol["dense2_b"], kind="vec", encoding=encoding)
        _set_layer(enc.mha.smolgen.ln2_gammas, smol["ln2_scale"], kind="vec", encoding=encoding)
        _set_layer(enc.mha.smolgen.ln2_betas, smol["ln2_bias"], kind="vec", encoding=encoding)
        _set_layer(enc.ln1_gammas, layer["ln1"]["scale"], kind="vec", encoding=encoding)
        _set_layer(enc.ln1_betas, layer["ln1"]["bias"], kind="vec", encoding=encoding)
        _set_layer(enc.ffn.dense1_w, layer["ffn"]["dense1_w"], kind="mat", encoding=encoding)
        _set_layer(enc.ffn.dense1_b, layer["ffn"]["dense1_b"], kind="vec", encoding=encoding)
        _set_layer(enc.ffn.dense2_w, layer["ffn"]["dense2_w"], kind="mat", encoding=encoding)
        _set_layer(enc.ffn.dense2_b, layer["ffn"]["dense2_b"], kind="vec", encoding=encoding)
        _set_layer(enc.ln2_gammas, layer["ln2"]["scale"], kind="vec", encoding=encoding)
        _set_layer(enc.ln2_betas, layer["ln2"]["bias"], kind="vec", encoding=encoding)

    pol = params["policy"]
    _set_layer(w.policy_heads.ip_pol_w, pol["dense1_w"], kind="mat", encoding=encoding)
    _set_layer(w.policy_heads.ip_pol_b, pol["dense1_b"], kind="vec", encoding=encoding)
    _set_layer(w.policy_heads.vanilla.ip2_pol_w, pol["q_w"], kind="mat", encoding=encoding)
    _set_layer(w.policy_heads.vanilla.ip2_pol_b, pol["q_b"], kind="vec", encoding=encoding)
    _set_layer(w.policy_heads.vanilla.ip3_pol_w, pol["k_w"], kind="mat", encoding=encoding)
    _set_layer(w.policy_heads.vanilla.ip3_pol_b, pol["k_b"], kind="vec", encoding=encoding)
    _set_layer(w.policy_heads.vanilla.ip4_pol_w, pol["prom_w"], kind="mat", encoding=encoding)
    if w.policy_heads.vanilla.HasField("pol_headcount"):
        w.policy_heads.vanilla.pol_headcount = int(params.get("headcount", w.policy_heads.vanilla.pol_headcount))

    val = params["value"]
    _set_layer(w.value_heads.winner.ip_val_w, val["embed_w"], kind="mat", encoding=encoding)
    _set_layer(w.value_heads.winner.ip_val_b, val["embed_b"], kind="vec", encoding=encoding)
    _set_layer(w.value_heads.winner.ip1_val_w, val["dense1_w"], kind="mat", encoding=encoding)
    _set_layer(w.value_heads.winner.ip1_val_b, val["dense1_b"], kind="vec", encoding=encoding)
    _set_layer(w.value_heads.winner.ip2_val_w, val["dense2_w"], kind="mat", encoding=encoding)
    _set_layer(w.value_heads.winner.ip2_val_b, val["dense2_b"], kind="vec", encoding=encoding)

    mov = params["moves_left"]
    _set_layer(w.ip_mov_w, mov["embed_w"], kind="mat", encoding=encoding)
    _set_layer(w.ip_mov_b, mov["embed_b"], kind="vec", encoding=encoding)
    _set_layer(w.ip1_mov_w, mov["dense1_w"], kind="mat", encoding=encoding)
    _set_layer(w.ip1_mov_b, mov["dense1_b"], kind="vec", encoding=encoding)
    _set_layer(w.ip2_mov_w, mov["dense2_w"], kind="mat", encoding=encoding)
    _set_layer(w.ip2_mov_b, mov["dense2_b"], kind="vec", encoding=encoding)

    _set_layer(w.smolgen_w, params["smolgen_w"], kind="mat", encoding=encoding)

    if w.policy_heads.policy_head_map:
        keep = [entry for entry in w.policy_heads.policy_head_map if entry.HasField("key") and entry.HasField("value")]
        if len(keep) != len(w.policy_heads.policy_head_map):
            del w.policy_heads.policy_head_map[:]
            w.policy_heads.policy_head_map.extend(keep)

    if w.value_heads.value_head_map:
        keep = [entry for entry in w.value_heads.value_head_map if entry.HasField("key") and entry.HasField("value")]
        if len(keep) != len(w.value_heads.value_head_map):
            del w.value_heads.value_head_map[:]
            w.value_heads.value_head_map.extend(keep)

    with gzip.open(out_path, "wb") as f:
        f.write(net.SerializeToString())

    return ExportStats(layers_written=1, encoding=encoding)
