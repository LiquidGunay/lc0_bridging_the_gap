"""Materialize rollout-pair JSONL and activation shards into solver-ready NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lc0jax.interpretability.pair_builders import (
    iter_rollout_pair_records,
    load_activation_index,
    materialize_rollout_differences,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-jsonl", required=True, help="Rollout-pair JSONL file.")
    parser.add_argument(
        "--activations",
        required=True,
        help="Activation shard directory or .npz file for trajectory FENs.",
    )
    parser.add_argument("--out", required=True, help="Output solver-ready pairs.npz.")
    parser.add_argument(
        "--activation-key",
        default="auto",
        choices=["auto", "token_activations", "embeddings"],
        help="Which activation array to use from shards.",
    )
    parser.add_argument("--mode", choices=["flat", "mean"], default="flat")
    parser.add_argument(
        "--index-mode",
        choices=["both", "single_even", "single_odd"],
        default="both",
        help="Which trajectory positions to aggregate before differencing.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    activation_index, key = load_activation_index(
        args.activations,
        activation_key=args.activation_key,
    )
    payload = materialize_rollout_differences(
        iter_rollout_pair_records(args.pairs_jsonl),
        activation_index,
        mode=args.mode,
        index_mode=args.index_mode,
        max_records=args.max_records,
    )
    metadata = {
        "pairs_jsonl": args.pairs_jsonl,
        "activations": args.activations,
        "activation_key": key,
        "mode": args.mode,
        "index_mode": args.index_mode,
        "max_records": args.max_records,
        "num_activation_fens": len(activation_index),
        "num_differences": int(payload["differences"].shape[0]),
        "dimension": int(payload["differences"].shape[1]),
        "records_consumed": int(payload["records_consumed"]),
        "records_or_lines_skipped": int(payload["records_or_lines_skipped"]),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **payload,
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )
    print(
        f"Wrote {out_path} with {metadata['num_differences']} differences "
        f"from {metadata['records_consumed']} records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
