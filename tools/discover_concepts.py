"""Discover concepts and optionally run activation patching."""

from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np

from lc0jax.interpretability.concepts import discover_concepts, patch_activations
from lc0jax.modeling.policy import attention_policy_map
from lc0jax.modeling.weights import load_pb_gz, map_bt4_weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-a", required=True)
    parser.add_argument("--embeddings-b", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="svm_cvxpy")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k", type=int, default=8, help="Number of directions for multi-vector methods")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--pb", default=None)
    parser.add_argument("--layer", default="trunk")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_embeddings(path: str, max_samples: int | None):
        p = Path(path)
        files = [p] if p.is_file() else sorted(p.glob("*.npz"))
        embeddings = []
        fens = []
        total = 0
        for file in files:
            data = np.load(file, allow_pickle=True)
            emb = data["embeddings"]
            embeddings.append(emb)
            if "fens" in data:
                fens.extend(data["fens"].tolist())
            total += emb.shape[0]
            if max_samples is not None and total >= max_samples:
                break
        emb_all = np.concatenate(embeddings, axis=0)
        if max_samples is not None:
            emb_all = emb_all[:max_samples]
            fens = fens[:max_samples]
        return emb_all, fens

    emb_a, fens_a = load_embeddings(args.embeddings_a, args.max_samples)
    emb_b, fens_b = load_embeddings(args.embeddings_b, args.max_samples)

    result = discover_concepts(emb_a, emb_b, method=args.method, k=args.k)
    direction = result["direction"]
    np.savez_compressed(out_dir / "concept_direction.npz", direction=direction)

    def score_and_write(vec, suffix: str):
        scores_a = emb_a @ vec
        scores_b = emb_b @ vec
        top_idx_a = np.argsort(scores_a)[-args.top_k:][::-1]
        top_idx_b = np.argsort(scores_b)[: args.top_k]

        with (out_dir / f"prototypes_a{suffix}.txt").open("w", encoding="utf-8") as f:
            for idx in top_idx_a:
                fen = fens_a[idx] if idx < len(fens_a) else ""
                f.write(f"{scores_a[idx]:.6f}\t{fen}\n")

        with (out_dir / f"prototypes_b{suffix}.txt").open("w", encoding="utf-8") as f:
            for idx in top_idx_b:
                fen = fens_b[idx] if idx < len(fens_b) else ""
                f.write(f"{scores_b[idx]:.6f}\t{fen}\n")

        return {
            "top_a": top_idx_a.tolist(),
            "top_b": top_idx_b.tolist(),
        }

    proto_info = {}
    if direction.ndim == 1:
        proto_info["vector_0"] = score_and_write(direction, "")
    else:
        for i in range(direction.shape[1]):
            proto_info[f"vector_{i}"] = score_and_write(direction[:, i], f"_{i:02d}")

    norm = result["norm"]
    if np.ndim(norm) == 0:
        norm_out = float(norm)
    else:
        norm_out = np.asarray(norm).tolist()

    report = {
        "method": result["method"],
        "norm": norm_out,
        "top_k": args.top_k,
        "samples_a": int(emb_a.shape[0]),
        "samples_b": int(emb_b.shape[0]),
        "vectors": proto_info,
    }
    if result.get("scores") is not None:
        report["scores"] = np.asarray(result["scores"]).tolist()

    if args.patch:
        if args.pb is None:
            raise ValueError("--pb is required when --patch is set.")
        bundle = load_pb_gz(args.pb)
        params = map_bt4_weights(bundle, mapping_table=attention_policy_map())
        patch_reports = []
        if direction.ndim == 1:
            directions = [(0, direction)]
        else:
            directions = list(enumerate(direction.T))
        for idx, vec in directions:
            sample_fen = None
            proto_key = f"vector_{idx}"
            if proto_key in proto_info and proto_info[proto_key]["top_a"]:
                sample_fen = fens_a[proto_info[proto_key]["top_a"][0]] if fens_a else None
            if sample_fen:
                delta = patch_activations(params, sample_fen, vec, alpha=args.alpha, layer=args.layer)
                patch_reports.append(
                    {
                        "vector": idx,
                        "sample_fen": sample_fen,
                        "delta_wdl": delta["delta_wdl"].squeeze().tolist(),
                    }
                )
        report["patch"] = patch_reports

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
