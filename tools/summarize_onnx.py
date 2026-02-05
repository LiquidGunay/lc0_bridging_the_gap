"""Summarize ONNX ops and tensor shapes."""

from __future__ import annotations

import argparse
from collections import Counter

import onnx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    counts = Counter(node.op_type for node in model.graph.node)
    print("Op counts:")
    for op, count in counts.most_common():
        print(f"  {op}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
