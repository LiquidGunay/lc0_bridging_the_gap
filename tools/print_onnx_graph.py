"""Print a simplified ONNX graph node list."""

from __future__ import annotations

import argparse

import onnx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    for idx, node in enumerate(model.graph.node):
        op = node.op_type
        name = node.name or f"node_{idx}"
        inputs = list(node.input)
        outputs = list(node.output)
        print(f"{idx:04d} {name} ({op})")
        print(f"  inputs: {inputs}")
        print(f"  outputs: {outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
