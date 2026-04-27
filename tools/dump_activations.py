"""Dump activation embeddings for a dataset."""

from __future__ import annotations

import argparse

from lc0jax.interpretability.activations import dump_activations
from lc0jax.interpretability.datasets import iter_activation_records, iter_fens
from lc0jax.modeling.policy import attention_policy_map
from lc0jax.modeling.weights import load_pb_gz, map_bt4_weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fens")
    input_group.add_argument("--records", help="JSONL records with fen/history_fens metadata.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", default="trunk")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--total-fens", type=int, default=None)
    parser.add_argument("--count-fens", action="store_true")
    parser.add_argument(
        "--activation-mode",
        choices=["mean", "flat"],
        default="mean",
        help=(
            "Projection stored as embeddings: mean pools 64 tokens, "
            "flat preserves square-local tokens."
        ),
    )
    parser.add_argument(
        "--store-token-activations",
        action="store_true",
        help=(
            "Also store raw [N, 64, channels] activations for later "
            "Schut-style rollout aggregation."
        ),
    )
    parser.add_argument(
        "--store-policy-logits",
        action="store_true",
        help=(
            "Also store policy logits. This is useful for prototype reports "
            "but increases shard size."
        ),
    )
    args = parser.parse_args()

    total_fens = args.total_fens
    if args.count_fens:
        input_path = args.records or args.fens
        with open(input_path, "r", encoding="utf-8") as handle:
            total_fens = sum(1 for _ in handle)
        print(f"Total FENs: {total_fens}")

    bundle = load_pb_gz(args.pb)
    params = map_bt4_weights(bundle, mapping_table=attention_policy_map())
    dataset_iter = iter_activation_records(args.records) if args.records else iter_fens(args.fens)
    dump_activations(
        params,
        dataset_iter,
        out_dir=args.out,
        layer=args.layer,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        progress_every=args.progress_every if args.progress_every > 0 else None,
        total_fens=total_fens,
        activation_mode=args.activation_mode,
        store_token_activations=args.store_token_activations,
        store_policy_logits=args.store_policy_logits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
