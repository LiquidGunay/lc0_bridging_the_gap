"""Compare ONNX oracle outputs to Flax outputs for a set of FENs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import onnxruntime as ort

from lc0jax.modeling.encode import encode_board
from lc0jax.modeling.inference import forward
from lc0jax.modeling.policy import attention_policy_map, legal_move_mask
from lc0jax.modeling.weights import load_pb_gz, map_bt4_weights

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--pb", required=True)
    parser.add_argument("--fens", required=True)
    args = parser.parse_args()

    if chess is None:
        raise ImportError("python-chess is required for comparing outputs.")

    fens_path = Path(args.fens)
    fens = [line.strip() for line in fens_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    planes = []
    boards = []
    for fen in fens:
        board = chess.Board(fen)
        boards.append(board)
        planes.append(encode_board(board, [], planes_layout="nchw", input_format="INPUT_CLASSICAL_112_PLANE"))
    planes = np.stack(planes, axis=0)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_outputs = sess.run(None, {input_name: planes.astype(np.float32)})
    output_names = [out.name for out in sess.get_outputs()]
    oracle = {name: value for name, value in zip(output_names, ort_outputs)}

    bundle = load_pb_gz(args.pb)
    params = map_bt4_weights(bundle, mapping_table=attention_policy_map())
    policy_logits, wdl, _mlh = forward(params, planes)

    policy_logits = np.asarray(policy_logits)
    wdl = np.asarray(wdl)
    if "/output/policy" in oracle:
        oracle_policy = oracle["/output/policy"]
    elif "output/policy" in oracle:
        oracle_policy = oracle["output/policy"]
    else:
        oracle_policy = oracle.get("policy")

    if "/output/wdl" in oracle:
        oracle_wdl = oracle["/output/wdl"]
    elif "output/wdl" in oracle:
        oracle_wdl = oracle["output/wdl"]
    else:
        oracle_wdl = oracle.get("wdl")
    if oracle_policy is None or oracle_wdl is None:
        raise ValueError(f"Unexpected ONNX outputs: {output_names}")

    top1_matches = 0
    top5_overlaps = []
    for idx, board in enumerate(boards):
        mask = legal_move_mask(board, "lc0_1858")
        mask = mask.astype(bool)
        oracle_masked = np.where(mask, oracle_policy[idx], -1e9)
        flax_masked = np.where(mask, policy_logits[idx], -1e9)
        if oracle_masked.argmax() == flax_masked.argmax():
            top1_matches += 1
        o_top5 = np.argpartition(oracle_masked, -5)[-5:]
        f_top5 = np.argpartition(flax_masked, -5)[-5:]
        overlap = len(set(o_top5) & set(f_top5)) / 5.0
        top5_overlaps.append(overlap)

    oracle_val = oracle_wdl[:, 0] - oracle_wdl[:, 2]
    flax_val = wdl[:, 0] - wdl[:, 2]
    value_corr = np.corrcoef(oracle_val, flax_val)[0, 1]

    print(f"Positions: {len(fens)}")
    print(f"Top-1 agreement: {top1_matches / max(1, len(fens)):.3f}")
    print(f"Top-5 overlap: {np.mean(top5_overlaps):.3f}")
    print(f"Value correlation: {value_corr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
