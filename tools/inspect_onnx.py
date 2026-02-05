"""Print ONNX input/output names, shapes, and initializers."""

from __future__ import annotations

import argparse

import onnx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    print("Inputs:")
    for inp in model.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {dims}")
    print("Outputs:")
    for out in model.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")
    print("Initializers:", len(model.graph.initializer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
