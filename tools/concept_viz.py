"""Generate HTML visualizations for concept prototypes."""

from __future__ import annotations

import argparse
import html
import json
import glob
from pathlib import Path

import chess
import chess.svg


def _read_lines(path: Path, *, top_n: int) -> list[tuple[float, str]]:
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            score = float(parts[0])
            fen = parts[1]
            items.append((score, fen))
            if len(items) >= top_n:
                break
    return items


def _render_board(fen: str) -> str:
    board = chess.Board(fen)
    return chess.svg.board(board=board, size=240)


def _vector_indices(run_dir: Path) -> list[int]:
    candidates = sorted(run_dir.glob("prototypes_a*.txt"))
    indices = []
    for path in candidates:
        name = path.stem
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


def _causal_map(run_dir: Path) -> dict[int, dict]:
    path = run_dir / "causal_report.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    vectors = report.get("vectors", [])
    return {int(item["vector"]): item for item in vectors}


def _render_run(run_dir: Path, *, top_n: int) -> str:
    report_path = run_dir / "report.json"
    report = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)

    method = report.get("method", "unknown")
    samples_a = report.get("samples_a", "n/a")
    samples_b = report.get("samples_b", "n/a")
    scores = report.get("scores")
    causal = _causal_map(run_dir)

    parts = [
        f"<section class='run'><h2>{html.escape(run_dir.name)}</h2>",
        f"<div class='meta'>method={html.escape(str(method))} | samples A={samples_a} B={samples_b}</div>",
    ]

    for idx in _vector_indices(run_dir):
        score = ""
        if isinstance(scores, list) and idx < len(scores):
            score = f" (score={scores[idx]:.6f})"
        parts.append(f"<h3>vector {idx}{score}</h3>")
        if idx in causal:
            entry = causal[idx]
            sig = entry.get("significant", False)
            sig_class = "significant" if sig else "nonsignificant"
            parts.append(
                "<div class='causal "
                + sig_class
                + "'>"
                + f"<span>mean Δvalue: {entry.get('mean_delta_value')}</span>"
                + f"<span>CI: [{entry.get('ci_low')}, {entry.get('ci_high')}]</span>"
                + f"<span>top1 change: {entry.get('top1_change_rate')}</span>"
                + f"<span>Δtop1 logit: {entry.get('mean_delta_top1_logit')}</span>"
                + f"<span>significant: {entry.get('significant')}</span>"
                + "</div>"
            )

        proto_a = run_dir / f"prototypes_a_{idx:02d}.txt"
        proto_b = run_dir / f"prototypes_b_{idx:02d}.txt"
        if not proto_a.exists():
            proto_a = run_dir / "prototypes_a.txt"
        if not proto_b.exists():
            proto_b = run_dir / "prototypes_b.txt"

        items_a = _read_lines(proto_a, top_n=top_n)
        items_b = _read_lines(proto_b, top_n=top_n)

        parts.append("<div class='grid'>")
        parts.append("<div class='column'><h4>Prototypes A</h4>")
        for score_val, fen in items_a:
            svg = _render_board(fen)
            parts.append(
                "<div class='card'>"
                f"<div class='score'>{score_val:.4f}</div>"
                f"{svg}"
                f"<div class='fen'>{html.escape(fen)}</div>"
                "</div>"
            )
        parts.append("</div>")

        parts.append("<div class='column'><h4>Prototypes B</h4>")
        for score_val, fen in items_b:
            svg = _render_board(fen)
            parts.append(
                "<div class='card'>"
                f"<div class='score'>{score_val:.4f}</div>"
                f"{svg}"
                f"<div class='fen'>{html.escape(fen)}</div>"
                "</div>"
            )
        parts.append("</div>")
        parts.append("</div>")

    parts.append("</section>")
    return "\n".join(parts)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", help="Concept run directories (default: data/concepts/*).")
    parser.add_argument("--out", default="data/concepts_viz.html")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    if args.runs:
        run_dirs = _collect_run_dirs(_expand_run_args(args.runs))
    else:
        run_dirs = sorted(Path("data/concepts").glob("*"))

    sections = []
    for run_dir in run_dirs:
        if run_dir.is_dir():
            sections.append(_render_run(run_dir, top_n=args.top_n))

    html_out = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LC0 Concepts</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .run {{ margin-bottom: 48px; }}
    .meta {{ color: #444; margin-bottom: 12px; }}
    .grid {{ display: flex; gap: 24px; flex-wrap: wrap; }}
    .column {{ flex: 1 1 420px; }}
    .card {{ border: 1px solid #ddd; padding: 8px; margin-bottom: 12px; background: #fafafa; }}
    .score {{ font-weight: bold; margin-bottom: 4px; }}
    .fen {{ font-family: monospace; font-size: 12px; word-break: break-word; }}
    .causal {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; font-size: 13px; }}
    .causal.significant {{ color: #0b5; font-weight: 600; }}
    .causal.nonsignificant {{ color: #a60; }}
    svg {{ display: block; margin: 4px 0; }}
  </style>
</head>
<body>
  <h1>LC0 Concept Visualizations</h1>
  {''.join(sections)}
</body>
</html>
"""

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
