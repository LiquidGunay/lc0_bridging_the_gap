"""Cluster dynamic rollout differences and solve sparse concept families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lc0jax.interpretability.concepts import dynamic_rollout_differences
from lc0jax.interpretability.dynamic_families import solve_dynamic_concept_families


def _load_differences(path: Path, *, mode: str, index_mode: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "differences" in data:
        differences = data["differences"]
    elif "optimal_rollouts" in data and "subpar_rollouts" in data:
        differences = dynamic_rollout_differences(
            data["optimal_rollouts"],
            data["subpar_rollouts"],
            mode=mode,
            index_mode=index_mode,
        )
    else:
        raise KeyError(
            "Pair file must contain either 'differences' or "
            "'optimal_rollouts' plus 'subpar_rollouts'"
        )
    differences = np.asarray(differences)
    if differences.ndim != 2:
        raise ValueError(f"Expected rank-2 differences after loading, got {differences.shape}")
    return differences


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _screening_report(family: dict[str, Any]) -> dict[str, Any]:
    report = {
        "enabled": bool(family.get("screening_enabled", False)),
        "method": family.get("screening_method"),
        "max_features": family.get("screening_max_features"),
        "screened_dimension": family.get("screened_dimension"),
    }
    indices = family.get("screening_indices")
    if indices is not None:
        report["indices_stored_in"] = "concept_direction.npz:screening_indices"
        report["selected_feature_preview"] = [int(value) for value in indices[:20]]
    return report


def _family_report(family: dict[str, Any], *, pairs: str, dimension: int) -> dict[str, Any]:
    return {
        "method": f"dynamic_concept_family_{family['method']}",
        "pairs": pairs,
        "family_id": family["family_id"],
        "cluster_id": family["cluster_id"],
        "num_pairs": family["num_rows"],
        "dimension": dimension,
        "row_indices_preview": [int(value) for value in family["row_indices"][:20]],
        "norm": family["norm"],
        "constraint_satisfaction": family["constraint_satisfaction"],
        "margin_satisfaction": family["margin_satisfaction"],
        "objective": family["objective"],
        "status": family["status"],
        "screening": _screening_report(family),
        "stability": family["stability"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="NPZ file with dynamic differences.")
    parser.add_argument("--out", required=True, help="Output family directory.")
    parser.add_argument("--mode", choices=["flat", "mean"], default="flat")
    parser.add_argument(
        "--index-mode",
        choices=["both", "single_even", "single_odd"],
        default="both",
    )
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument(
        "--max-features",
        type=int,
        default=2048,
        help="Screening cap per cluster; set <=0 to solve exact full-dimensional families.",
    )
    parser.add_argument("--screening-method", choices=["abs_mean", "mean_abs"], default="abs_mean")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-normalize-rows", action="store_true")
    parser.add_argument(
        "--bootstrap-count",
        type=int,
        default=0,
        help="Bootstrap resamples per family; each resample runs another sparse solve.",
    )
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--bootstrap-min-rows", type=int, default=4)
    parser.add_argument("--bootstrap-cosine-threshold", type=float, default=0.7)
    args = parser.parse_args()

    differences = _load_differences(Path(args.pairs), mode=args.mode, index_mode=args.index_mode)
    max_features = None if args.max_features <= 0 else args.max_features
    result = solve_dynamic_concept_families(
        differences,
        n_clusters=args.clusters,
        min_cluster_size=args.min_cluster_size,
        max_features=max_features,
        screening_method=args.screening_method,
        c=args.c,
        margin=args.margin,
        standardize=not args.no_standardize,
        seed=args.seed,
        normalize_rows=not args.no_normalize_rows,
        bootstrap_count=args.bootstrap_count,
        bootstrap_fraction=args.bootstrap_fraction,
        bootstrap_min_rows=args.bootstrap_min_rows,
        bootstrap_cosine_threshold=args.bootstrap_cosine_threshold,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "families.npz",
        labels=result["labels"],
        centers=result["centers"],
        family_ids=result["family_ids"],
        cluster_ids=result["cluster_ids"],
        directions=result["directions"],
        raw_directions=result["raw_directions"],
        row_indices=result["row_indices"],
    )

    family_reports = []
    for family, direction, raw_direction in zip(
        result["families"],
        result["directions"],
        result["raw_directions"],
    ):
        family_dir = out_dir / f"family_{family['family_id']:03d}"
        family_dir.mkdir(parents=True, exist_ok=True)
        direction_payload = {
            "direction": direction,
            "raw_direction": raw_direction,
        }
        if family.get("screening_indices") is not None:
            direction_payload["screening_indices"] = family["screening_indices"]
            direction_payload["screening_scores"] = family["screening_scores"]
        np.savez_compressed(family_dir / "concept_direction.npz", **direction_payload)
        report = _family_report(family, pairs=args.pairs, dimension=int(differences.shape[1]))
        (family_dir / "report.json").write_text(
            json.dumps(_json_value(report), indent=2) + "\n",
            encoding="utf-8",
        )
        family_reports.append(report)

    report = {
        "method": "dynamic_concept_families",
        "pairs": args.pairs,
        "mode": args.mode,
        "index_mode": args.index_mode,
        "num_pairs": int(differences.shape[0]),
        "dimension": int(differences.shape[1]),
        "clusters_requested": args.clusters,
        "families_solved": len(result["families"]),
        "min_cluster_size": args.min_cluster_size,
        "max_features": max_features,
        "screening_method": args.screening_method,
        "standardize": not args.no_standardize,
        "normalize_rows": not args.no_normalize_rows,
        "bootstrap_count": args.bootstrap_count,
        "skipped_clusters": result["skipped_clusters"],
        "families": family_reports,
        "artifact": "families.npz",
    }
    (out_dir / "report.json").write_text(
        json.dumps(_json_value(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(result['families'])} dynamic concept families to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
