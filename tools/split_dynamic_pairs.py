"""Split solver-ready dynamic rollout pairs into train/test NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lc0jax.interpretability.dynamic_splits import (
    infer_pair_row_count,
    root_split_summary,
    split_pair_indices,
    subset_pairs_payload,
)


def _load_pairs(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _decode_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    array = np.asarray(value)
    if array.shape != ():
        return {}
    raw = array.item()
    if raw in (None, ""):
        return {}
    try:
        decoded = json.loads(str(raw))
    except json.JSONDecodeError:
        return {"source_metadata_raw": str(raw)}
    return decoded if isinstance(decoded, dict) else {"source_metadata": decoded}


def _with_split_metadata(
    payload: dict[str, np.ndarray],
    *,
    split_name: str,
    source_pairs: Path,
    test_fraction: float,
    seed: int,
    summary: dict[str, int],
    extra_row_aligned_keys: set[str],
) -> dict[str, np.ndarray]:
    out = dict(payload)
    metadata = _decode_metadata(payload.get("metadata"))
    if "num_differences" in metadata:
        metadata["source_num_differences"] = metadata["num_differences"]
    metadata["num_differences"] = (
        summary["num_train_rows"] if split_name == "train" else summary["num_test_rows"]
    )
    metadata["split"] = {
        "name": split_name,
        "source_pairs": str(source_pairs),
        "group_key": "root_fens_without_fullmove",
        "test_fraction": float(test_fraction),
        "seed": int(seed),
        "extra_row_aligned_keys": sorted(extra_row_aligned_keys),
        **summary,
    }
    out["metadata"] = np.asarray(json.dumps(metadata, indent=2), dtype=object)
    return out


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Source solver-ready pairs.npz file.")
    parser.add_argument("--out-train", required=True, help="Output train pairs.npz file.")
    parser.add_argument("--out-test", required=True, help="Output held-out test pairs.npz file.")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--row-aligned-key",
        action="append",
        default=[],
        help="Additional payload key to subset row-wise; may be repeated.",
    )
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    payload = _load_pairs(pairs_path)
    if "root_fens" not in payload:
        raise KeyError("pairs.npz must contain root_fens for grouped held-out splits")

    row_count = infer_pair_row_count(payload)
    root_fens = payload["root_fens"]
    root_array = np.asarray(root_fens)
    if root_array.ndim == 0 or root_array.shape[0] != row_count:
        raise ValueError("root_fens length must match the number of difference rows")

    train_indices, test_indices = split_pair_indices(
        root_fens,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    summary = root_split_summary(root_fens, train_indices, test_indices)
    extra_row_keys = set(args.row_aligned_key)
    train_payload = subset_pairs_payload(
        payload,
        train_indices,
        row_count=row_count,
        row_aligned_keys=None if not extra_row_keys else extra_row_keys,
    )
    test_payload = subset_pairs_payload(
        payload,
        test_indices,
        row_count=row_count,
        row_aligned_keys=None if not extra_row_keys else extra_row_keys,
    )
    train_payload = _with_split_metadata(
        train_payload,
        split_name="train",
        source_pairs=pairs_path,
        test_fraction=args.test_fraction,
        seed=args.seed,
        summary=summary,
        extra_row_aligned_keys=extra_row_keys,
    )
    test_payload = _with_split_metadata(
        test_payload,
        split_name="test",
        source_pairs=pairs_path,
        test_fraction=args.test_fraction,
        seed=args.seed,
        summary=summary,
        extra_row_aligned_keys=extra_row_keys,
    )

    _write_npz(Path(args.out_train), train_payload)
    _write_npz(Path(args.out_test), test_payload)
    print(
        "Wrote train/test splits with "
        f"{summary['num_train_rows']} train rows and {summary['num_test_rows']} test rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
