"""Causal validation for concept directions via activation patching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lc0jax.modeling.encode import encode_board
from lc0jax.modeling.inference import forward
from lc0jax.modeling.policy import attention_policy_map
from lc0jax.modeling.weights import load_pb_gz, map_bt4_weights

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None


def _expected_value(wdl: np.ndarray) -> np.ndarray:
    """Convert WDL probabilities to a scalar value in [-1, 1]."""
    return wdl[..., 0] - wdl[..., 2]


def _bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    if n_boot <= 0 or values.size == 0:
        return float("nan"), float("nan")
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample = rng.integers(0, n, size=n)
        means[idx] = values[sample].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _load_vectors(path: Path) -> tuple[np.ndarray, list[int]]:
    data = np.load(path, allow_pickle=True)
    direction = data["direction"]
    if direction.ndim == 1:
        vectors = direction[:, None]
    else:
        vectors = direction
    indices = list(range(vectors.shape[1]))
    return vectors, indices


def _parse_indices(text: str | None, max_len: int) -> list[int]:
    if not text:
        return list(range(max_len))
    indices = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        indices.append(int(token))
    return indices


def _load_fens_from_embeddings(path: Path, *, max_samples: int | None) -> list[str]:
    files = [path] if path.is_file() else sorted(path.glob("*.npz"))
    fens: list[str] = []
    for file in files:
        data = np.load(file, allow_pickle=True)
        if "fens" not in data:
            continue
        fens.extend(data["fens"].tolist())
        if max_samples is not None and len(fens) >= max_samples:
            break
    if max_samples is not None:
        fens = fens[:max_samples]
    return fens


def _load_fens_from_file(path: Path, *, max_samples: int | None) -> list[str]:
    fens: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fen = line.strip()
            if not fen:
                continue
            fens.append(fen)
            if max_samples is not None and len(fens) >= max_samples:
                break
    return fens


def _batch(items: list[str], batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", required=True, help="Concept directory or concept_direction.npz file.")
    parser.add_argument("--pb", required=True, help="Path to BT4 .pb.gz weights.")
    parser.add_argument("--embeddings", help="Embeddings dir or .npz file with stored FENs.")
    parser.add_argument("--fens", help="Path to a newline-delimited FEN file.")
    parser.add_argument("--out", help="Output JSON path (default: <concept>/causal_report.json).")
    parser.add_argument("--layer", default="trunk")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vectors", help="Comma-separated vector indices to evaluate.")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    if chess is None:
        raise ImportError("python-chess is required for causal validation.")

    concept_path = Path(args.concept)
    if concept_path.is_dir():
        concept_file = concept_path / "concept_direction.npz"
    else:
        concept_file = concept_path
        concept_path = concept_path.parent
    if not concept_file.exists():
        raise FileNotFoundError(f"Missing concept direction file: {concept_file}")

    if args.embeddings is None and args.fens is None:
        raise ValueError("Provide either --embeddings or --fens for sampling.")

    if args.embeddings:
        fens = _load_fens_from_embeddings(Path(args.embeddings), max_samples=None)
    else:
        fens = _load_fens_from_file(Path(args.fens), max_samples=None)

    if not fens:
        raise ValueError("No FENs loaded for causal validation.")

    rng = np.random.default_rng(args.seed)
    if args.max_samples is not None and len(fens) > args.max_samples:
        indices = rng.choice(len(fens), size=args.max_samples, replace=False)
        fens = [fens[idx] for idx in indices]

    vectors, all_indices = _load_vectors(concept_file)
    chosen_indices = _parse_indices(args.vectors, len(all_indices))

    bundle = load_pb_gz(args.pb)
    params = map_bt4_weights(bundle, mapping_table=attention_policy_map())

    results = []
    for idx in chosen_indices:
        vec = vectors[:, idx]
        delta_values = []
        delta_top_logits = []
        top1_changes = []

        for batch in _batch(fens, args.batch_size):
            planes = []
            for fen in batch:
                board = chess.Board(fen)
                planes.append(
                    encode_board(board, [], planes_layout="nchw", input_format="INPUT_CLASSICAL_112_PLANE")
                )
            planes = np.stack(planes, axis=0)

            base_policy, base_wdl, _mlh = forward(params, planes)
            patched_policy, patched_wdl, _mlh = forward(
                params,
                planes,
                patch={"layer": args.layer, "vector": vec, "alpha": args.alpha},
            )

            base_policy = np.asarray(base_policy)
            patched_policy = np.asarray(patched_policy)
            base_wdl = np.asarray(base_wdl)
            patched_wdl = np.asarray(patched_wdl)

            base_value = _expected_value(base_wdl)
            patched_value = _expected_value(patched_wdl)
            delta_values.append(patched_value - base_value)

            base_top = np.argmax(base_policy, axis=1)
            patched_top = np.argmax(patched_policy, axis=1)
            top1_changes.append((base_top != patched_top).astype(np.float32))

            base_top_logit = np.take_along_axis(base_policy, base_top[:, None], axis=1).squeeze()
            patched_top_logit = np.take_along_axis(patched_policy, base_top[:, None], axis=1).squeeze()
            delta_top_logits.append(patched_top_logit - base_top_logit)

        delta_values = np.concatenate(delta_values, axis=0)
        delta_top_logits = np.concatenate(delta_top_logits, axis=0)
        top1_changes = np.concatenate(top1_changes, axis=0)

        ci_low, ci_high = _bootstrap_ci(delta_values, rng=rng, n_boot=args.bootstrap)
        results.append(
            {
                "vector": int(idx),
                "num_samples": int(delta_values.shape[0]),
                "mean_delta_value": float(delta_values.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_delta_top1_logit": float(delta_top_logits.mean()),
                "top1_change_rate": float(top1_changes.mean()),
                "significant": bool(ci_low > 0 or ci_high < 0),
            }
        )

    report = {
        "concept": str(concept_file),
        "pb": str(args.pb),
        "layer": args.layer,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "vectors": results,
    }

    out_path = Path(args.out) if args.out else (concept_path / "causal_report.json")
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
