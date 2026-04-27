"""Filter concept vectors with Schut-style machine-vs-human novelty curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lc0jax.interpretability.novelty import novelty_curve


def _load_embeddings(path: Path, *, max_samples: int | None) -> np.ndarray:
    files = [path] if path.is_file() else sorted(path.glob("*.npz"))
    arrays = []
    total = 0
    for file in files:
        data = np.load(file, allow_pickle=True)
        embeddings = data["embeddings"]
        if embeddings.ndim == 3:
            embeddings = embeddings.reshape((embeddings.shape[0], -1))
        arrays.append(embeddings)
        total += embeddings.shape[0]
        if max_samples is not None and total >= max_samples:
            break
    if not arrays:
        raise RuntimeError(f"No embedding shards found at {path}")
    merged = np.concatenate(arrays, axis=0)
    if max_samples is not None:
        merged = merged[:max_samples]
    return merged


def _load_directions(path: Path) -> tuple[np.ndarray, Path]:
    concept_file = path / "concept_direction.npz" if path.is_dir() else path
    data = np.load(concept_file, allow_pickle=True)
    directions = data["direction"]
    if directions.ndim == 1:
        directions = directions[:, None]
    return directions, concept_file


def _parse_ranks(text: str) -> list[int]:
    ranks = []
    for token in text.split(","):
        token = token.strip()
        if token:
            ranks.append(int(token))
    if not ranks:
        raise ValueError("At least one rank is required")
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept",
        required=True,
        help="Concept directory or concept_direction.npz file.",
    )
    parser.add_argument(
        "--machine-embeddings",
        required=True,
        help="LC0/self-play activation shard or directory.",
    )
    parser.add_argument(
        "--human-embeddings",
        required=True,
        help="Human activation shard or directory.",
    )
    parser.add_argument(
        "--out",
        help="Output JSON path; defaults to <concept>/novelty_report.json.",
    )
    parser.add_argument("--ranks", default="32,64,128,256,512,1024")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--accept-positive-fraction",
        type=float,
        default=0.8,
        help="Accept vectors whose novelty is positive at this fraction of evaluated ranks.",
    )
    args = parser.parse_args()

    concept_path = Path(args.concept)
    directions, concept_file = _load_directions(concept_path)
    machine = _load_embeddings(Path(args.machine_embeddings), max_samples=args.max_samples)
    human = _load_embeddings(Path(args.human_embeddings), max_samples=args.max_samples)
    ranks = _parse_ranks(args.ranks)

    curves = novelty_curve(directions, machine, human, ranks=ranks)
    accepted = [
        item["vector"]
        for item in curves
        if item["novelty_area"] > 0
        and item["positive_rank_fraction"] >= args.accept_positive_fraction
    ]

    report = {
        "concept": str(concept_file),
        "machine_embeddings": str(args.machine_embeddings),
        "human_embeddings": str(args.human_embeddings),
        "machine_samples": int(machine.shape[0]),
        "human_samples": int(human.shape[0]),
        "ranks": ranks,
        "accept_positive_fraction": args.accept_positive_fraction,
        "accepted_vectors": accepted,
        "vectors": curves,
    }

    if args.out:
        out_path = Path(args.out)
    elif concept_path.is_dir():
        out_path = concept_path / "novelty_report.json"
    else:
        out_path = concept_file.with_name("novelty_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Novelty report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
