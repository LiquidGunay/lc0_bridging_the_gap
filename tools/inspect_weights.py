"""Inspect LC0 .pb.gz weights metadata and layer counts."""

from __future__ import annotations

import argparse

from lc0jax.modeling.weights import load_pb_gz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb", required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    bundle = load_pb_gz(args.pb)
    meta = bundle.metadata
    print("Has weights:", meta.get("has_weights"))
    print("Has onnx:", meta.get("has_onnx"))
    print("Weights encoding:", meta.get("weights_encoding"))
    if meta.get("min_version"):
        v = meta["min_version"]
        print("Min version:", f"{v['major']}.{v['minor']}.{v['patch']}")
    if meta.get("network_format"):
        nf = meta["network_format"]
        try:
            from lc0jax.proto import net_pb2  # type: ignore

            print("Input format:", net_pb2.NetworkFormat.InputFormat.Name(nf.get("input")))
            print("Output format:", net_pb2.NetworkFormat.OutputFormat.Name(nf.get("output")))
            print("Network format:", net_pb2.NetworkFormat.NetworkStructure.Name(nf.get("network")))
            print("Policy format:", net_pb2.NetworkFormat.PolicyFormat.Name(nf.get("policy")))
            print("Value format:", net_pb2.NetworkFormat.ValueFormat.Name(nf.get("value")))
        except Exception:
            print("Input format:", nf.get("input"))
            print("Output format:", nf.get("output"))
            print("Network format:", nf.get("network"))
            print("Policy format:", nf.get("policy"))
            print("Value format:", nf.get("value"))
    print("Tensor count:", len(bundle.tensors))

    try:
        from lc0jax.proto import net_pb2  # type: ignore
        import gzip

        net = net_pb2.Net()
        with gzip.open(args.pb, "rb") as f:
            net.ParseFromString(f.read())
        if net.HasField("weights"):
            w = net.weights
            print("Encoder blocks:", len(w.encoder))
            print("Head count:", w.headcount)
            print("Policy encoder blocks:", len(w.pol_encoder))
            print("Policy head count:", w.pol_headcount)
    except Exception:
        pass
    if args.list:
        for key in sorted(bundle.tensors.keys()):
            print(key, bundle.tensors[key].shape, bundle.tensors[key].dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
