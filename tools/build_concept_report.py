"""Build a markdown concept report from concept run directories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import glob


def _expand_run_args(raw: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for item in raw:
        if any(ch in item for ch in "*?[]"):
            expanded.extend(Path(match) for match in sorted(glob.glob(item)))
        else:
            expanded.append(Path(item))
    return expanded


def _collect_run_dirs(inputs: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for path in inputs:
        if path.is_file() and path.name == "report.json":
            run_dirs.append(path.parent)
            continue
        if path.is_dir():
            if (path / "report.json").exists() or list(path.glob("prototypes_a*.txt")):
                run_dirs.append(path)
                continue
            for child in sorted(path.iterdir()):
                if not child.is_dir():
                    continue
                if (child / "report.json").exists() or list(child.glob("prototypes_a*.txt")):
                    run_dirs.append(child)
    seen = set()
    deduped = []
    for run_dir in run_dirs:
        key = run_dir.resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(run_dir)
    return deduped


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_top_lines(path: Path, *, top_n: int) -> list[str]:
    if not path.exists():
        return []
    lines = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= top_n:
                break
    return lines


def _vector_indices(run_dir: Path) -> list[int]:
    candidates = sorted(run_dir.glob("prototypes_a*.txt"))
    indices = []
    for path in candidates:
        name = path.stem  # prototypes_a or prototypes_a_00
        if name == "prototypes_a":
            indices.append(0)
        else:
            suffix = name.split("_")[-1]
            try:
                indices.append(int(suffix))
            except ValueError:
                continue
    if not indices:
        indices = [0]
    return sorted(set(indices))


def _patch_map(report: dict) -> dict[int, dict]:
    patch = report.get("patch")
    if isinstance(patch, list):
        return {int(item["vector"]): item for item in patch}
    if isinstance(patch, dict):
        return {0: patch}
    return {}


def _causal_map(run_dir: Path) -> dict[int, dict]:
    path = run_dir / "causal_report.json"
    if not path.exists():
        return {}
    report = _read_json(path)
    vectors = report.get("vectors", [])
    return {int(item["vector"]): item for item in vectors}


def _format_section(run_dir: Path, *, top_n: int) -> list[str]:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return []
    report = _read_json(report_path)
    patch_map = _patch_map(report)
    causal_map = _causal_map(run_dir)

    method = report.get("method", "unknown")
    samples_a = report.get("samples_a", "n/a")
    samples_b = report.get("samples_b", "n/a")
    norm = report.get("norm", "n/a")
    scores = report.get("scores")

    lines = [
        f"## {run_dir.name}",
        f"- method: {method}",
        f"- samples: A={samples_a} B={samples_b}",
        f"- norm: {norm}",
    ]

    vector_ids = _vector_indices(run_dir)
    for idx in vector_ids:
        score = None
        if isinstance(scores, list) and idx < len(scores):
            score = scores[idx]
        header = f"### vector {idx}"
        if score is not None:
            header += f" (score={score:.6f})"
        lines.append(header)

        proto_a = run_dir / f"prototypes_a_{idx:02d}.txt"
        proto_b = run_dir / f"prototypes_b_{idx:02d}.txt"
        if not proto_a.exists():
            proto_a = run_dir / "prototypes_a.txt"
        if not proto_b.exists():
            proto_b = run_dir / "prototypes_b.txt"
        top_a = _read_top_lines(proto_a, top_n=top_n)
        top_b = _read_top_lines(proto_b, top_n=top_n)

        lines.append("Prototypes A (score\\tFEN):")
        lines.extend([f"- {line}" for line in top_a] or ["- (none)"])
        lines.append("Prototypes B (score\\tFEN):")
        lines.extend([f"- {line}" for line in top_b] or ["- (none)"])

        patch = patch_map.get(idx)
        if patch:
            lines.append(f"- patch sample FEN: {patch.get('sample_fen', '')}")
            if "delta_wdl" in patch:
                lines.append(f"- patch delta WDL: {patch['delta_wdl']}")

        causal = causal_map.get(idx)
        if causal:
            lines.append("- causal validation:")
            lines.append(f"  - mean delta value: {causal.get('mean_delta_value')}")
            lines.append(f"  - 95% CI: [{causal.get('ci_low')}, {causal.get('ci_high')}]")
            lines.append(f"  - top1 change rate: {causal.get('top1_change_rate')}")
            lines.append(f"  - mean delta top1 logit: {causal.get('mean_delta_top1_logit')}")
            lines.append(f"  - significant: {causal.get('significant')}")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", help="Concept run directories (default: data/concepts/*).")
    parser.add_argument("--out", default="data/concept_report.md")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    if args.runs:
        run_dirs = _collect_run_dirs(_expand_run_args(args.runs))
    else:
        run_dirs = sorted(Path("data/concepts").glob("*"))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    lines = [
        "# Concept Report",
        "",
        f"Generated: {stamp}",
        "",
    ]

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        section = _format_section(run_dir, top_n=args.top_n)
        if section:
            lines.extend(section)
            lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
