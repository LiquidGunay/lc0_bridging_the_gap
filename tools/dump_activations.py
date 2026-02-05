"""Dump activation embeddings for a dataset."""

from __future__ import annotations

import argparse

from lc0jax.interpretability.activations import dump_activations
from lc0jax.interpretability.datasets import iter_fens
from lc0jax.modeling.policy import attention_policy_map
from lc0jax.modeling.weights import load_pb_gz, map_bt4_weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb", required=True)
    parser.add_argument("--fens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", default="trunk")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--total-fens", type=int, default=None)
    parser.add_argument("--count-fens", action="store_true")
    args = parser.parse_args()

    total_fens = args.total_fens
    if args.count_fens:
        with open(args.fens, "r", encoding="utf-8") as handle:
            total_fens = sum(1 for _ in handle)
        print(f"Total FENs: {total_fens}")

    bundle = load_pb_gz(args.pb)
    params = map_bt4_weights(bundle, mapping_table=attention_policy_map())
    dump_activations(
        params,
        iter_fens(args.fens),
        out_dir=args.out,
        layer=args.layer,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        progress_every=args.progress_every if args.progress_every > 0 else None,
        total_fens=total_fens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
