"""Evaluate prototype teachability with a low-rank policy adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lc0jax.interpretability.dynamic_teachability import teachability_lift_report


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(payload)
    return records


def _parse_indices(text: str | None) -> np.ndarray | None:
    if text in (None, ""):
        return None
    path = Path(text)
    if path.exists():
        values = [
            int(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return np.asarray(values, dtype=np.int64)
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    return np.asarray(values, dtype=np.int64)


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if key not in data:
        available = ", ".join(sorted(data.files))
        raise KeyError(f"{path} does not contain key '{key}'. Available keys: {available}")
    return np.asarray(data[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="NPZ file with frozen feature rows.")
    parser.add_argument(
        "--teacher-logits",
        help="Optional NPZ file with teacher policy logits. Defaults to --features.",
    )
    parser.add_argument(
        "--eval-features",
        help="Optional held-out NPZ file with frozen feature rows for evaluation.",
    )
    parser.add_argument(
        "--eval-teacher-logits",
        help=(
            "Optional held-out NPZ file with teacher policy logits. Defaults to "
            "--eval-features when provided."
        ),
    )
    parser.add_argument("--curriculum", required=True, help="Teachability curriculum JSONL.")
    parser.add_argument("--out", required=True, help="Output teachability report JSON.")
    parser.add_argument("--feature-key", default="differences")
    parser.add_argument("--logits-key", default="policy_logits")
    parser.add_argument("--eval-feature-key", default=None)
    parser.add_argument("--eval-logits-key", default=None)
    parser.add_argument(
        "--eval-indices",
        help="Optional comma-separated row indices or newline-delimited index file.",
    )
    parser.add_argument("--max-prototypes", type=int, default=None)
    parser.add_argument("--max-controls", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    features_path = Path(args.features)
    logits_path = Path(args.teacher_logits) if args.teacher_logits else features_path
    features = _load_npz_array(features_path, args.feature_key)
    teacher_logits = _load_npz_array(logits_path, args.logits_key)
    eval_features = None
    eval_teacher_logits = None
    eval_features_path = None
    eval_logits_path = None
    if args.eval_features:
        eval_features_path = Path(args.eval_features)
        eval_logits_path = (
            Path(args.eval_teacher_logits) if args.eval_teacher_logits else eval_features_path
        )
        eval_features = _load_npz_array(
            eval_features_path,
            args.eval_feature_key or args.feature_key,
        )
        eval_teacher_logits = _load_npz_array(
            eval_logits_path,
            args.eval_logits_key or args.logits_key,
        )
    records = _read_jsonl(Path(args.curriculum))
    report = teachability_lift_report(
        features,
        teacher_logits,
        records,
        eval_features=eval_features,
        eval_teacher_logits=eval_teacher_logits,
        eval_indices=_parse_indices(args.eval_indices),
        max_prototypes=args.max_prototypes,
        max_controls=args.max_controls,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
        temperature=args.temperature,
        seed=args.seed,
    )
    report.update(
        {
            "features": str(features_path),
            "teacher_logits": str(logits_path),
            "feature_key": args.feature_key,
            "logits_key": args.logits_key,
            "eval_features": None if eval_features_path is None else str(eval_features_path),
            "eval_teacher_logits": None if eval_logits_path is None else str(eval_logits_path),
            "eval_feature_key": args.eval_feature_key or args.feature_key,
            "eval_logits_key": args.eval_logits_key or args.logits_key,
            "curriculum": args.curriculum,
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_value(report), indent=2) + "\n", encoding="utf-8")
    print(f"Teachability report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
