"""Run the ONNX oracle on a list of FENs or pre-encoded planes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lc0jax.modeling.encode import encode_board
from lc0jax.uci.oracle import run_onnx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--fens", help="Path to newline-delimited FENs")
    parser.add_argument("--planes-npy", help="Path to a .npy array of planes")
    parser.add_argument("--dummy-batch", type=int, default=0, help="Run with zero planes")
    parser.add_argument(
        "--input-format",
        default="INPUT_CLASSICAL_112_PLANE",
        help="LC0 input format string",
    )
    args = parser.parse_args()

    if not args.fens and not args.planes_npy and args.dummy_batch <= 0:
        raise SystemExit("Provide --fens, --planes-npy, or --dummy-batch.")

    if args.planes_npy:
        planes = np.load(args.planes_npy)
        if planes.ndim == 3:
            planes = planes[None, ...]
    elif args.dummy_batch > 0:
        # Default to LC0-like input plane count; update once encoder is implemented.
        planes = np.zeros((args.dummy_batch, 112, 8, 8), dtype=np.float32)
    else:
        fens_path = Path(args.fens)
        fens = [line.strip() for line in fens_path.read_text().splitlines() if line.strip()]
        planes_list = []
        for fen in fens:
            # TODO: implement proper history handling; current call expects encoder to be implemented.
            planes_list.append(
                encode_board(
                    fen,
                    history=[],
                    planes_layout="nchw",
                    input_format=args.input_format,
                )
            )
        planes = np.stack(planes_list, axis=0)

    outputs = run_onnx(args.onnx, planes)
    for name, value in outputs.items():
        print(f"{name}: shape={value.shape} dtype={value.dtype}")
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
