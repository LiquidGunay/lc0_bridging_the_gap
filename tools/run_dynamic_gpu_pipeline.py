"""Run a GPU-oriented Schut-style LC0 dynamic concept pipeline.

This script intentionally composes the existing single-purpose tools instead
of reimplementing MCTS extraction, activation dumping, pair materialization, or
concept solving. It adds reproducible run directories, GPU-aware environment
defaults, PGN/FEN root preparation, sharding, resume behavior, and command
metadata around those tools.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import chess.pgn

from lc0jax.interpretability.datasets import filter_fens
from lc0jax.interpretability.manifests import dynamic_roots_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = "models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
STAGES = ["prepare", "mcts", "activations", "materialize", "split", "sweep"]


def _utc_run_id() -> str:
    now = _dt.datetime.now(tz=_dt.UTC)
    return f"dynamic_gpu_{now:%Y%m%d_%H%M%S}"


def _stage_index(stage: str) -> int:
    return STAGES.index(stage)


def _runs_stage(stop_after: str, stage: str) -> bool:
    return _stage_index(stage) <= _stage_index(stop_after)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def _shard_label(args: argparse.Namespace) -> str:
    return f"shard_{args.shard_index:03d}_of_{args.shard_count:03d}"


def _work_dir_for_run(run_dir: Path, args: argparse.Namespace) -> Path:
    if args.shard_count <= 1:
        return run_dir
    return run_dir / "shards" / _shard_label(args)


def _root_history_summary(args: argparse.Namespace, *, input_mode: str) -> dict[str, Any]:
    has_fen_sources = bool(args.fens)
    has_pgn_sources = bool(args.pgn)
    if input_mode == "root_records" and has_pgn_sources and not has_fen_sources:
        history_mode = "pgn_pre_root"
        root_history_complete = True
    elif input_mode == "root_records" and has_pgn_sources and has_fen_sources:
        history_mode = "mixed_pgn_and_root_only_fen"
        root_history_complete = False
    else:
        history_mode = "root_only_fen"
        root_history_complete = False
    return {
        "source_counts": {
            "fen_input_paths": len(args.fens),
            "pgn_input_paths": len(args.pgn),
        },
        "history_mode": history_mode,
        "root_history_complete": root_history_complete,
        "contains_history_poor_roots": not root_history_complete,
    }


def _stage_marker(work_dir: Path, stage: str) -> Path:
    return work_dir / "_stage_status" / f"{stage}.done.json"


def _write_stage_marker(path: Path, *, stage: str, outputs: list[Path]) -> None:
    _json_dump(
        path,
        {
            "stage": stage,
            "completed_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(),
            "outputs": [str(output) for output in outputs],
        },
    )


def _clean_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _iter_fens_from_pgn(path: Path, *, ply_stride: int):
    with path.open("r", encoding="utf-8", errors="ignore") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            ply_idx = 0
            for move in game.mainline_moves():
                board.push(move)
                if ply_idx % ply_stride == 0:
                    yield board.fen()
                ply_idx += 1


def _iter_records_from_pgn(path: Path, *, ply_stride: int, history_len: int):
    with path.open("r", encoding="utf-8", errors="ignore") as pgn:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            history = [board.fen()]
            game_id = game.headers.get("Site") or game.headers.get("Event") or f"game_{game_idx}"
            ply_idx = 0
            for move in game.mainline_moves():
                board.push(move)
                history.append(board.fen())
                if ply_idx % ply_stride == 0:
                    ply = board.ply()
                    yield {
                        "fen": board.fen(),
                        "history_fens": history[-history_len:],
                        "game_id": game_id,
                        "game_index": game_idx,
                        "ply": ply,
                        "source": str(path),
                        "record_id": f"{path.name}:game_{game_idx:08d}:ply_{ply:04d}",
                    }
                ply_idx += 1
            game_idx += 1


def _write_raw_candidates(
    *,
    fens_paths: list[Path],
    pgn_paths: list[Path],
    out_path: Path,
    max_positions: int | None,
    ply_stride: int,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for fen_path in fens_paths:
            with fen_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    fen = line.strip()
                    if not fen:
                        continue
                    out.write(fen + "\n")
                    written += 1
                    if max_positions is not None and written >= max_positions:
                        return written
        for pgn_path in pgn_paths:
            for fen in _iter_fens_from_pgn(pgn_path, ply_stride=ply_stride):
                out.write(fen + "\n")
                written += 1
                if max_positions is not None and written >= max_positions:
                    return written
    return written


def _write_raw_candidate_records(
    *,
    fens_paths: list[Path],
    pgn_paths: list[Path],
    out_records: Path,
    out_fens: Path,
    max_positions: int | None,
    ply_stride: int,
    history_len: int,
) -> int:
    out_records.parent.mkdir(parents=True, exist_ok=True)
    out_fens.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_records.open("w", encoding="utf-8") as records_out:
        with out_fens.open("w", encoding="utf-8") as fens_out:
            for fen_path in fens_paths:
                with fen_path.open("r", encoding="utf-8") as handle:
                    for line_idx, line in enumerate(handle):
                        fen = line.strip()
                        if not fen:
                            continue
                        record = {
                            "fen": fen,
                            "history_fens": [fen],
                            "source": str(fen_path),
                            "record_id": f"{fen_path.name}:line_{line_idx:08d}",
                        }
                        records_out.write(json.dumps(record, sort_keys=True) + "\n")
                        fens_out.write(fen + "\n")
                        written += 1
                        if max_positions is not None and written >= max_positions:
                            return written
            for pgn_path in pgn_paths:
                for record in _iter_records_from_pgn(
                    pgn_path,
                    ply_stride=ply_stride,
                    history_len=history_len,
                ):
                    records_out.write(json.dumps(record, sort_keys=True) + "\n")
                    fens_out.write(record["fen"] + "\n")
                    written += 1
                    if max_positions is not None and written >= max_positions:
                        return written
    return written


def _filter_records_by_fens(records_path: Path, fens_path: Path, out_path: Path) -> int:
    keep_counts: Counter[str] = Counter()
    with fens_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fen = line.strip()
            if fen:
                keep_counts[fen] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with records_path.open("r", encoding="utf-8") as inp:
        with out_path.open("w", encoding="utf-8") as out:
            for line in inp:
                if not line.strip():
                    continue
                record = json.loads(line)
                fen = str(record.get("fen", ""))
                if keep_counts[fen] <= 0:
                    continue
                out.write(json.dumps(record, sort_keys=True) + "\n")
                keep_counts[fen] -= 1
                kept += 1
    return kept


def _write_shard(src: Path, dst: Path, *, shard_count: int, shard_index: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src.open("r", encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line_idx, line in enumerate(inp):
            fen = line.strip()
            if not fen:
                continue
            if line_idx % shard_count != shard_index:
                continue
            out.write(fen + "\n")
            written += 1
    return written


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.executable,
        "platform": sys.platform,
        "uv_cache_dir": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv")),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "nvidia_device_files": [
            path
            for path in ("/dev/nvidia0", "/dev/nvidiactl", "/proc/driver/nvidia/version")
            if Path(path).exists()
        ],
    }
    try:
        import jax

        devices = jax.devices()
        info.update(
            {
                "jax_version": jax.__version__,
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in devices],
                "jax_gpu_visible": any(
                    device.platform in {"gpu", "cuda"} for device in devices
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive for broken envs
        info.update({"jax_error": f"{type(exc).__name__}: {exc}", "jax_gpu_visible": False})
    return info


def resolve_lc0_backend(args: argparse.Namespace, runtime: dict[str, Any]) -> str | None:
    backend = args.lc0_backend
    if backend in {"none", ""}:
        return None
    if backend != "auto":
        return backend
    if args.prefer_gpu and runtime.get("jax_gpu_visible"):
        return args.auto_lc0_gpu_backend
    return None


def _subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = args.uv_cache_dir
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if args.jax_preallocate else "false"
    if args.jax_platform != "auto":
        env["JAX_PLATFORM_NAME"] = args.jax_platform
    return env


def _visible_env(env: dict[str, str]) -> dict[str, str]:
    keys = ["UV_CACHE_DIR", "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_PLATFORM_NAME"]
    return {key: env[key] for key in keys if key in env}


def _run_command(
    *,
    stage: str,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
    marker_path: Path,
    marker_outputs: list[Path],
) -> None:
    record = {
        "stage": stage,
        "cmd": cmd,
        "cwd": str(REPO_ROOT),
        "env": _visible_env(env),
        "dry_run": dry_run,
        "started_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(),
    }
    print(f"[{stage}] {' '.join(cmd)}", flush=True)
    if dry_run:
        record.update({"returncode": None, "elapsed_seconds": 0.0})
        _append_jsonl(log_path, record)
        return

    start = time.time()
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    record.update({"returncode": completed.returncode, "elapsed_seconds": time.time() - start})
    _append_jsonl(log_path, record)
    if completed.returncode != 0:
        raise RuntimeError(f"Stage {stage} failed with exit code {completed.returncode}")
    missing = [path for path in marker_outputs if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Stage {stage} completed but did not create expected outputs: "
            + ", ".join(str(path) for path in missing)
        )
    _write_stage_marker(marker_path, stage=stage, outputs=marker_outputs)


def _skip_stage(args: argparse.Namespace, outputs: list[Path], marker_path: Path) -> bool:
    return bool(
        args.resume
        and marker_path.exists()
        and outputs
        and all(path.exists() for path in outputs)
    )


def _prepare_roots(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    use_root_records = bool(args.pgn)
    raw_roots = run_dir / "candidate_roots.raw.fens"
    filtered_roots = run_dir / "candidate_roots.filtered.fens"
    raw_records = run_dir / "candidate_roots.raw.records.jsonl"
    filtered_records = run_dir / "candidate_roots.filtered.records.jsonl"
    shard_roots = run_dir / (
        f"candidate_roots.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.fens"
    )
    shard_records = run_dir / (
        f"candidate_roots.shard_{args.shard_index:03d}_of_{args.shard_count:03d}.records.jsonl"
    )
    if use_root_records:
        roots_path = shard_records if args.shard_count > 1 else filtered_records
        roots_arg = "--root-records"
        input_mode = "root_records"
    else:
        roots_path = shard_roots if args.shard_count > 1 else filtered_roots
        roots_arg = "--fens"
        input_mode = "fens"

    if args.resume and roots_path.exists():
        return {
            "input_mode": input_mode,
            "root_arg": roots_arg,
            "raw_roots": raw_roots,
            "filtered_roots": filtered_roots,
            "raw_records": raw_records if use_root_records else None,
            "filtered_records": filtered_records if use_root_records else None,
            "roots": roots_path,
            "raw_count": _line_count(raw_roots) if raw_roots.exists() else None,
            "filtered_count": _line_count(filtered_roots) if filtered_roots.exists() else None,
            "root_count": _line_count(roots_path),
            "skipped": True,
            **_root_history_summary(args, input_mode=input_mode),
        }

    fens_paths = [Path(path) for path in args.fens]
    pgn_paths = [Path(path) for path in args.pgn]
    if use_root_records:
        raw_count = _write_raw_candidate_records(
            fens_paths=fens_paths,
            pgn_paths=pgn_paths,
            out_records=raw_records,
            out_fens=raw_roots,
            max_positions=args.max_candidate_positions,
            ply_stride=args.ply_stride,
            history_len=args.history_len,
        )
    else:
        raw_count = _write_raw_candidates(
            fens_paths=fens_paths,
            pgn_paths=pgn_paths,
            out_path=raw_roots,
            max_positions=args.max_candidate_positions,
            ply_stride=args.ply_stride,
        )
    filtered_count = filter_fens(
        str(raw_roots),
        out_fens=str(filtered_roots),
        max_positions=args.max_roots,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        min_phase=args.min_phase,
        max_phase=args.max_phase,
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
        min_nonpawn=args.min_nonpawn,
        max_nonpawn=args.max_nonpawn,
        dedupe=args.dedupe,
        progress_every=args.prepare_progress_every,
        progress_label="Prepared roots",
    )
    if use_root_records:
        filtered_count = _filter_records_by_fens(raw_records, filtered_roots, filtered_records)
        if args.shard_count > 1:
            root_count = _write_shard(
                filtered_records,
                shard_records,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
            )
        else:
            root_count = filtered_count
    elif args.shard_count > 1:
        root_count = _write_shard(
            filtered_roots,
            shard_roots,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
    else:
        root_count = filtered_count

    return {
        "input_mode": input_mode,
        "root_arg": roots_arg,
        "raw_roots": raw_roots,
        "filtered_roots": filtered_roots,
        "raw_records": raw_records if use_root_records else None,
        "filtered_records": filtered_records if use_root_records else None,
        "roots": roots_path,
        "raw_count": raw_count,
        "filtered_count": filtered_count,
        "root_count": root_count,
        "skipped": False,
        **_root_history_summary(args, input_mode=input_mode),
    }


def _metadata_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['run_id']}",
        "",
        f"- created_utc: {summary['created_utc']}",
        f"- run_dir: `{summary['run_dir']}`",
        f"- work_dir: `{summary['work_dir']}`",
        f"- shard: `{summary['shard']}`",
        f"- stop_after: `{summary['stop_after']}`",
        f"- dry_run: {summary['dry_run']}",
        f"- resume: {summary['resume']}",
        f"- python: `{summary['runtime'].get('python')}`",
        f"- uv_cache_dir: `{summary['runtime'].get('uv_cache_dir')}`",
        f"- jax_default_backend: `{summary['runtime'].get('jax_default_backend', 'n/a')}`",
        f"- jax_devices: `{summary['runtime'].get('jax_devices', [])}`",
        f"- lc0_backend_resolved: `{summary.get('lc0_backend_resolved')}`",
        f"- weights: `{summary['weights']}`",
        f"- root_input_mode: `{summary['roots'].get('input_mode')}`",
        f"- roots: `{summary['roots'].get('roots')}`",
        f"- root_count: {summary['roots'].get('root_count')}",
        f"- nodes: {summary['mcts'].get('nodes')}",
        f"- multipv: {summary['mcts'].get('multipv')}",
        f"- max_pairs: {summary['mcts'].get('max_pairs')}",
        f"- dynamic_roots_manifest: `{summary['outputs'].get('dynamic_roots_manifest')}`",
        "",
        "Command log: `commands.jsonl`",
        "",
    ]
    return "\n".join(lines)


def _run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        status = "planned"
    elif args.stop_after != STAGES[-1]:
        status = "partial"
    else:
        status = "completed"
    return {
        "status": status,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "stop_after": args.stop_after,
        "stages_requested": [stage for stage in STAGES if _runs_stage(args.stop_after, stage)],
    }


def _output_status(outputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    status = {}
    for key, value in outputs.items():
        path = Path(value)
        status[key] = {"path": path, "exists": path.exists()}
    return status


def _filter_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_candidate_positions": args.max_candidate_positions,
        "max_roots": args.max_roots,
        "ply_stride": args.ply_stride,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "min_phase": args.min_phase,
        "max_phase": args.max_phase,
        "min_pieces": args.min_pieces,
        "max_pieces": args.max_pieces,
        "min_nonpawn": args.min_nonpawn,
        "max_nonpawn": args.max_nonpawn,
        "dedupe": args.dedupe,
    }


def _search_manifest(args: argparse.Namespace, *, resolved_threads: int) -> dict[str, Any]:
    return {
        "nodes": args.nodes,
        "movetime_ms": args.movetime_ms,
        "multipv": args.multipv,
        "max_pairs": args.max_pairs,
        "max_depth": args.max_depth,
        "min_delta_cp": args.min_delta_cp,
        "max_delta_cp": args.max_delta_cp,
        "threads": resolved_threads,
        "history_len": args.history_len,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare high-strength PGN/FEN roots and run the dynamic LC0 concept "
            "pipeline with GPU-friendly defaults."
        )
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None, help="Explicit output directory.")
    parser.add_argument("--out-root", default="data/runs")
    parser.add_argument(
        "--fens",
        action="extend",
        nargs="+",
        default=[],
        help="Input root FEN list. Accepts one or more paths per flag.",
    )
    parser.add_argument(
        "--pgn",
        action="extend",
        nargs="+",
        default=[],
        help="Input PGN file. Accepts one or more paths per flag.",
    )
    parser.add_argument("--max-candidate-positions", type=int, default=None)
    parser.add_argument("--max-roots", type=int, default=None)
    parser.add_argument("--ply-stride", type=int, default=1)
    parser.add_argument("--min-ply", type=int, default=12)
    parser.add_argument("--max-ply", type=int, default=None)
    parser.add_argument("--min-phase", type=float, default=0.2)
    parser.add_argument("--max-phase", type=float, default=0.95)
    parser.add_argument("--min-pieces", type=int, default=12)
    parser.add_argument("--max-pieces", type=int, default=None)
    parser.add_argument("--min-nonpawn", type=int, default=4)
    parser.add_argument("--max-nonpawn", type=int, default=None)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-progress-every", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)

    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--lc0",
        default=os.environ.get("LC0_BIN", "/tmp/lc0-src/build/release/lc0"),
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--nodes", type=int, default=800)
    parser.add_argument("--movetime-ms", type=int, default=None)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=4000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-delta-cp", type=int, default=25)
    parser.add_argument("--max-delta-cp", type=int, default=None)
    parser.add_argument("--threads", type=int, default=0, help="0 chooses up to 12 local threads.")
    parser.add_argument(
        "--lc0-backend",
        default="auto",
        help="LC0 backend to configure: auto, none, cuda, cudnn, eigen, etc.",
    )
    parser.add_argument(
        "--auto-lc0-gpu-backend",
        default="cuda",
        help="Backend used when --lc0-backend=auto and a GPU is visible.",
    )
    parser.add_argument("--backend-opts", default=None)
    parser.add_argument("--prefer-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--uci-timeout", type=float, default=120.0)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--skip-errors", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--activation-batch-size", type=int, default=32)
    parser.add_argument("--activation-shard-size", type=int, default=4096)
    parser.add_argument("--activation-layer", default="trunk")
    parser.add_argument("--store-policy-logits", action="store_true")
    parser.add_argument("--jax-platform", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--jax-preallocate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--uv-cache-dir",
        default=os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv")),
        help="Shared uv cache directory to avoid duplicate wheel downloads.",
    )

    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-features", default="2048")
    parser.add_argument("--screening-methods", default="abs_mean")
    parser.add_argument("--policy-margin", action="store_true")
    parser.add_argument("--policy-alphas", default="0.1,0.3,1.0,3.0")
    parser.add_argument("--policy-direction-keys", default="direction")
    parser.add_argument("--policy-max-pairs", type=int, default=64)
    parser.add_argument("--policy-batch-size", type=int, default=8)

    parser.add_argument("--stop-after", choices=STAGES, default="sweep")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runtime-check-only", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.runtime_check_only:
        return
    if not args.fens and not args.pgn:
        parser.error("provide at least one --fens or --pgn input")
    if args.shard_count < 1:
        parser.error("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < shard-count")
    if args.ply_stride < 1:
        parser.error("--ply-stride must be >= 1")
    if _runs_stage(args.stop_after, "mcts") and not args.dry_run:
        lc0_path = Path(args.lc0)
        if not lc0_path.exists():
            parser.error(f"--lc0 does not exist: {lc0_path}")
        weights = Path(args.weights)
        if not weights.exists():
            parser.error(f"--weights does not exist: {weights}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    runtime = _runtime_info()
    if args.runtime_check_only:
        print(json.dumps(runtime, indent=2, sort_keys=True))
        return 0

    run_id = args.run_id or _utc_run_id()
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.out_root) / run_id
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = _work_dir_for_run(run_dir, args)
    work_dir.mkdir(parents=True, exist_ok=True)

    commands_log = work_dir / "commands.jsonl"
    if not args.resume and commands_log.exists():
        commands_log.unlink()

    env = _subprocess_env(args)
    roots = _prepare_roots(args, work_dir)
    if _runs_stage(args.stop_after, "mcts") and roots.get("root_count") == 0:
        raise RuntimeError("No candidate root positions remained after filtering.")

    mcts_dir = work_dir / "mcts_pairs"
    activations_dir = work_dir / "activations" / "trajectory_flat"
    concepts_dir = work_dir / "concepts" / "screening_sweep"
    pairs_jsonl = mcts_dir / "pairs.jsonl"
    trajectory_records = mcts_dir / "trajectory.records.jsonl"
    trajectory_fens = mcts_dir / "trajectory.fens"
    pairs_npz = mcts_dir / "pairs.npz"
    train_npz = mcts_dir / "pairs.train.npz"
    test_npz = mcts_dir / "pairs.test.npz"
    resolved_threads = args.threads or min(os.cpu_count() or 1, 12)
    lc0_backend = resolve_lc0_backend(args, runtime)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": _dt.datetime.now(tz=_dt.UTC).isoformat(),
        "run_dir": run_dir,
        "work_dir": work_dir,
        "shard": {
            "enabled": args.shard_count > 1,
            "count": args.shard_count,
            "index": args.shard_index,
            "label": _shard_label(args) if args.shard_count > 1 else None,
        },
        "stop_after": args.stop_after,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "runtime": runtime,
        "inputs": {"fens": args.fens, "pgn": args.pgn},
        "weights": args.weights,
        "lc0": args.lc0,
        "lc0_backend_resolved": lc0_backend,
        "roots": roots,
        "mcts": {
            "nodes": args.nodes,
            "movetime_ms": args.movetime_ms,
            "multipv": args.multipv,
            "max_pairs": args.max_pairs,
            "threads": resolved_threads,
        },
        "outputs": {
            "pairs_jsonl": pairs_jsonl,
            "trajectory_records": trajectory_records,
            "trajectory_fens": trajectory_fens,
            "activations": activations_dir,
            "pairs_npz": pairs_npz,
            "train_pairs": train_npz,
            "test_pairs": test_npz,
            "screening_sweep": concepts_dir,
        },
    }

    mcts_outputs = [pairs_jsonl, trajectory_records]
    if _runs_stage(args.stop_after, "mcts") and not _skip_stage(
        args, mcts_outputs, _stage_marker(work_dir, "mcts")
    ):
        if not args.resume and not args.dry_run:
            _clean_paths([mcts_dir])
        cmd = [
            args.python,
            "tools/build_mcts_pairs.py",
            roots["root_arg"],
            str(roots["roots"]),
            "--out-jsonl",
            str(pairs_jsonl),
            "--out-trajectory-records",
            str(trajectory_records),
            "--out-trajectory-fens",
            str(trajectory_fens),
            "--lc0",
            args.lc0,
            "--weights",
            args.weights,
            "--nodes",
            str(args.nodes),
            "--multipv",
            str(args.multipv),
            "--max-depth",
            str(args.max_depth),
            "--max-pairs",
            str(args.max_pairs),
            "--min-delta-cp",
            str(args.min_delta_cp),
            "--threads",
            str(resolved_threads),
            "--uci-timeout",
            str(args.uci_timeout),
            "--history-len",
            str(args.history_len),
            "--progress-every",
            "25",
        ]
        if args.movetime_ms is not None:
            cmd.extend(["--movetime-ms", str(args.movetime_ms)])
        if args.max_delta_cp is not None:
            cmd.extend(["--max-delta-cp", str(args.max_delta_cp)])
        if lc0_backend:
            cmd.extend(["--backend", lc0_backend])
        if args.backend_opts:
            cmd.extend(["--backend-opts", args.backend_opts])
        if args.skip_errors:
            cmd.append("--skip-errors")
        _run_command(
            stage="mcts",
            cmd=cmd,
            env=env,
            log_path=commands_log,
            dry_run=args.dry_run,
            marker_path=_stage_marker(work_dir, "mcts"),
            marker_outputs=mcts_outputs,
        )

    activation_outputs = [activations_dir / "done.txt", activations_dir / "shard_0000.npz"]
    if _runs_stage(args.stop_after, "activations") and not _skip_stage(
        args, activation_outputs, _stage_marker(work_dir, "activations")
    ):
        if not args.resume and not args.dry_run:
            _clean_paths([activations_dir])
        cmd = [
            args.python,
            "tools/dump_activations.py",
            "--pb",
            args.weights,
            "--records",
            str(trajectory_records),
            "--out",
            str(activations_dir),
            "--layer",
            args.activation_layer,
            "--activation-mode",
            "flat",
            "--store-token-activations",
            "--batch-size",
            str(args.activation_batch_size),
            "--shard-size",
            str(args.activation_shard_size),
            "--progress-every",
            "1000",
            "--count-fens",
        ]
        if args.store_policy_logits:
            cmd.append("--store-policy-logits")
        _run_command(
            stage="activations",
            cmd=cmd,
            env=env,
            log_path=commands_log,
            dry_run=args.dry_run,
            marker_path=_stage_marker(work_dir, "activations"),
            marker_outputs=activation_outputs,
        )

    materialize_outputs = [pairs_npz]
    if _runs_stage(args.stop_after, "materialize") and not _skip_stage(
        args, materialize_outputs, _stage_marker(work_dir, "materialize")
    ):
        if not args.resume and not args.dry_run:
            _clean_paths(materialize_outputs)
        cmd = [
            args.python,
            "tools/materialize_mcts_pairs.py",
            "--pairs-jsonl",
            str(pairs_jsonl),
            "--activations",
            str(activations_dir),
            "--out",
            str(pairs_npz),
            "--mode",
            "flat",
        ]
        if args.store_policy_logits:
            cmd.extend(["--policy-logits-key", "policy_logits"])
        _run_command(
            stage="materialize",
            cmd=cmd,
            env=env,
            log_path=commands_log,
            dry_run=args.dry_run,
            marker_path=_stage_marker(work_dir, "materialize"),
            marker_outputs=materialize_outputs,
        )

    split_outputs = [train_npz, test_npz]
    if _runs_stage(args.stop_after, "split") and not _skip_stage(
        args, split_outputs, _stage_marker(work_dir, "split")
    ):
        if not args.resume and not args.dry_run:
            _clean_paths(split_outputs)
        cmd = [
            args.python,
            "tools/split_dynamic_pairs.py",
            "--pairs",
            str(pairs_npz),
            "--out-train",
            str(train_npz),
            "--out-test",
            str(test_npz),
            "--test-fraction",
            str(args.test_fraction),
            "--seed",
            str(args.split_seed),
        ]
        _run_command(
            stage="split",
            cmd=cmd,
            env=env,
            log_path=commands_log,
            dry_run=args.dry_run,
            marker_path=_stage_marker(work_dir, "split"),
            marker_outputs=split_outputs,
        )

    sweep_outputs = [concepts_dir / "summary.json"]
    if _runs_stage(args.stop_after, "sweep") and not _skip_stage(
        args, sweep_outputs, _stage_marker(work_dir, "sweep")
    ):
        if not args.resume and not args.dry_run:
            _clean_paths([concepts_dir])
        cmd = [
            args.python,
            "tools/sweep_dynamic_screening.py",
            "--train-pairs",
            str(train_npz),
            "--test-pairs",
            str(test_npz),
            "--out",
            str(concepts_dir),
            "--mode",
            "flat",
            "--max-features",
            args.max_features,
            "--screening-methods",
            args.screening_methods,
            "--random-count",
            "128",
            "--shuffled-label-count",
            "128",
            "--top-k",
            "32",
            "--prototype-random-count",
            "32",
        ]
        if args.policy_margin:
            cmd.extend(
                [
                    "--pb",
                    args.weights,
                    "--policy-alphas",
                    args.policy_alphas,
                    "--policy-direction-keys",
                    args.policy_direction_keys,
                    "--policy-max-pairs",
                    str(args.policy_max_pairs),
                    "--policy-batch-size",
                    str(args.policy_batch_size),
                ]
            )
        else:
            cmd.append("--skip-policy-margin")
        _run_command(
            stage="sweep",
            cmd=cmd,
            env=env,
            log_path=commands_log,
            dry_run=args.dry_run,
            marker_path=_stage_marker(work_dir, "sweep"),
            marker_outputs=sweep_outputs,
        )

    manifest_path = work_dir / "dynamic_roots_manifest.json"
    summary["outputs"]["dynamic_roots_manifest"] = manifest_path
    output_status = _output_status(summary["outputs"])
    output_status["dynamic_roots_manifest"]["exists"] = True
    manifest = dynamic_roots_manifest(
        run_id=run_id,
        created_utc=summary["created_utc"],
        run=_run_manifest(args),
        inputs=summary["inputs"],
        roots=roots,
        filters=_filter_manifest(args),
        search=_search_manifest(args, resolved_threads=resolved_threads),
        model={"weights": args.weights},
        lc0={
            "binary": args.lc0,
            "backend": lc0_backend,
            "backend_opts": args.backend_opts,
        },
        outputs=summary["outputs"],
        output_status=output_status,
    )
    _json_dump(manifest_path, manifest)
    _json_dump(work_dir / "run_summary.json", summary)
    (work_dir / "RUN_METADATA.md").write_text(_metadata_markdown(summary), encoding="utf-8")
    print(f"Run directory: {work_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
