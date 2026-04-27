"""Build Schut-style LC0 optimal-vs-subpar rollout-pair records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess.engine

from lc0jax.interpretability.mcts_rollouts import (
    activation_records_for_line,
    build_rollout_pair_record,
)


def _iter_fens(path: Path, *, start_line: int = 0):
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx < start_line:
                continue
            fen = line.strip()
            if fen:
                yield fen


def _configure_engine(args) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci([args.lc0], timeout=args.uci_timeout)
    options = {}
    if args.weights:
        options["WeightsFile"] = args.weights
    if args.threads:
        options["Threads"] = args.threads
    if args.backend:
        options["Backend"] = args.backend
    if args.backend_opts:
        options["BackendOptions"] = args.backend_opts
    if args.multipv_option:
        options["MultiPV"] = args.multipv
    if options:
        engine.configure(options)
    return engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens", required=True, help="Input root FEN list.")
    parser.add_argument("--out-jsonl", required=True, help="Output rollout-pair JSONL path.")
    parser.add_argument(
        "--out-trajectory-fens",
        help="Optional newline-delimited unique FENs from all selected best/subpar PVs.",
    )
    parser.add_argument(
        "--out-trajectory-records",
        help=(
            "Optional JSONL activation records from all selected PVs. Prefer this "
            "over --out-trajectory-fens for 112-plane LC0 inputs because it "
            "preserves rolling history."
        ),
    )
    parser.add_argument("--lc0", required=True, help="Path to LC0 binary.")
    parser.add_argument("--weights", default=None, help="Path to LC0 network weights.")
    parser.add_argument("--nodes", type=int, default=800)
    parser.add_argument("--movetime-ms", type=int, default=None)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument(
        "--multipv-option",
        action="store_true",
        help=(
            "Also configure the UCI MultiPV option; LC0 usually accepts "
            "multipv on analyse directly."
        ),
    )
    parser.add_argument("--min-delta-cp", type=int, default=25)
    parser.add_argument("--max-delta-cp", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--start-line", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--backend-opts", default=None)
    parser.add_argument("--uci-timeout", type=float, default=60.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--history-len",
        type=int,
        default=8,
        help="Maximum number of FENs to keep in each trajectory activation record.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help=(
            "Continue after malformed FENs or per-position engine errors. "
            "By default these errors fail the run."
        ),
    )
    args = parser.parse_args()

    if args.nodes is None and args.movetime_ms is None:
        raise ValueError("Provide --nodes or --movetime-ms (or both).")

    limit = chess.engine.Limit(
        nodes=args.nodes,
        time=None if args.movetime_ms is None else args.movetime_ms / 1000.0,
    )

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_mode = "a" if args.append else "w"
    records_mode = "a" if args.append else "w"
    trajectory_fens: set[str] = set()
    seen = 0
    kept = 0
    failed = 0

    trajectory_records_handle = None
    if args.out_trajectory_records:
        out_records = Path(args.out_trajectory_records)
        out_records.parent.mkdir(parents=True, exist_ok=True)
        trajectory_records_handle = out_records.open(records_mode, encoding="utf-8")

    engine = _configure_engine(args)
    try:
        with out_jsonl.open(out_mode, encoding="utf-8") as handle:
            for fen in _iter_fens(Path(args.fens), start_line=args.start_line):
                seen += 1
                source_line = args.start_line + seen - 1
                try:
                    record = build_rollout_pair_record(
                        engine,
                        fen,
                        limit,
                        multipv=args.multipv,
                        min_delta_cp=args.min_delta_cp,
                        max_delta_cp=args.max_delta_cp,
                        max_depth=args.max_depth,
                        node_budget=args.nodes,
                    )
                except (ValueError, chess.engine.EngineError, TimeoutError) as exc:
                    failed += 1
                    print(
                        f"Failed root line {source_line}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not args.skip_errors:
                        raise
                    continue
                if record is None:
                    continue
                if trajectory_records_handle is not None:
                    root_id = f"root_{source_line:08d}"
                    for item in activation_records_for_line(
                        record.best,
                        line_id=f"{root_id}:best",
                        history_len=args.history_len,
                    ):
                        trajectory_records_handle.write(json.dumps(item) + "\n")
                    for line_idx, line in enumerate(record.subpar):
                        for item in activation_records_for_line(
                            line,
                            line_id=f"{root_id}:subpar_{line_idx:02d}",
                            history_len=args.history_len,
                        ):
                            trajectory_records_handle.write(json.dumps(item) + "\n")
                handle.write(json.dumps(record.to_json()) + "\n")
                kept += 1
                if args.out_trajectory_fens:
                    trajectory_fens.update(record.best.fens)
                    for line in record.subpar:
                        trajectory_fens.update(line.fens)
                if args.progress_every and kept % args.progress_every == 0:
                    print(f"Seen {seen} roots, kept {kept} rollout pairs", flush=True)
                if args.max_pairs is not None and kept >= args.max_pairs:
                    break
    finally:
        if trajectory_records_handle is not None:
            trajectory_records_handle.close()
        engine.quit()

    if args.out_trajectory_fens:
        out_fens = Path(args.out_trajectory_fens)
        out_fens.parent.mkdir(parents=True, exist_ok=True)
        out_fens.write_text("\n".join(sorted(trajectory_fens)) + "\n", encoding="utf-8")

    print(f"MCTS rollout pairs kept: {kept} (seen {seen})")
    if failed:
        print(f"MCTS rollout roots failed: {failed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
