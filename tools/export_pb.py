"""Export JAX BT4 parameters to LC0 .pb.gz weights."""

from __future__ import annotations

import argparse

from lc0jax.modeling.policy import attention_policy_map
from lc0jax.modeling.weights import load_pb_gz, map_bt4_weights
from lc0jax.uci.export import export_bt4_params


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb", required=True, help="Source .pb.gz (used to load params)")
    parser.add_argument("--out", required=True, help="Output .pb.gz")
    parser.add_argument(
        "--template",
        default=None,
        help="Template .pb.gz to copy metadata from (defaults to --pb)",
    )
    parser.add_argument(
        "--encoding",
        default="LINEAR16",
        choices=["LINEAR16", "FLOAT16", "BFLOAT16"],
    )
    args = parser.parse_args()

    bundle = load_pb_gz(args.pb)
    params = map_bt4_weights(bundle, mapping_table=attention_policy_map())
    export_bt4_params(
        params,
        template_path=args.template or args.pb,
        out_path=args.out,
        encoding=args.encoding,
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
