"""Download helper for Lichess PGNs and LC0 self-play training data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
import urllib.request

from lc0jax.interpretability.datasets import (
    filter_pgn_stream,
    latest_lichess_standard_filename,
    parse_lichess_standard_index,
    parse_sha256_sums,
)
from lc0jax.training.downloads import parse_training_index, pick_latest_training_tars


LICHESS_STANDARD_INDEX = "https://database.lichess.org/standard/"
LICHESS_STANDARD_SHA256 = "https://database.lichess.org/standard/sha256sums.txt"
LICHESS_BROADCAST_API = "https://lichess.org/api/broadcast"
LICHESS_BROADCAST_TOP = "https://lichess.org/api/broadcast/top"
LICHESS_PUZZLE_URL = "https://database.lichess.org/puzzles/lichess_db_puzzle.csv.zst"
LC0_TRAINING_INDEX = "https://storage.lczero.org/files/training_data/"


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "lc0jax-downloader/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, out_path: str, *, expected_sha256: str | None = None, force: bool = False) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not force:
        if expected_sha256:
            actual = _sha256_file(out_path)
            if actual == expected_sha256:
                print(f"Already downloaded: {out_path}")
                return out_path
            print("Checksum mismatch, re-downloading.")
        else:
            print(f"Already downloaded: {out_path}")
            return out_path

    tmp_path = out_path + ".partial"
    req = urllib.request.Request(url, headers={"User-Agent": "lc0jax-downloader/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out:
        total = int(resp.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100.0 * downloaded / total
                print(f"\rDownloaded {downloaded}/{total} bytes ({pct:0.1f}%)", end="")
        if total:
            print()

    os.replace(tmp_path, out_path)
    if expected_sha256:
        actual = _sha256_file(out_path)
        if actual != expected_sha256:
            raise RuntimeError(f"SHA256 mismatch for {out_path}: {actual} != {expected_sha256}")
    return out_path


def decompress_zst(zst_path: str, out_path: str, *, force: bool = False) -> str:
    if os.path.exists(out_path) and not force:
        print(f"Already decompressed: {out_path}")
        return out_path
    try:
        import zstandard as zstd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("zstandard is required to decompress .zst files.") from exc

    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as src, open(out_path, "wb") as dst:
        with dctx.stream_reader(src) as reader:
            shutil.copyfileobj(reader, dst, length=1024 * 1024)
    return out_path


def extract_tar(tar_path: str, out_dir: str) -> None:
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(out_dir)


def cmd_lichess(args: argparse.Namespace) -> int:
    index_html = fetch_text(LICHESS_STANDARD_INDEX)
    entries = parse_lichess_standard_index(index_html)
    if not entries:
        raise RuntimeError("No entries found in Lichess standard index.")

    if args.month:
        filename = f"lichess_db_standard_rated_{args.month}.pgn.zst"
    else:
        filename = latest_lichess_standard_filename(index_html)
        if filename is None:
            raise RuntimeError("Failed to determine latest Lichess standard PGN.")

    sha256_map = parse_sha256_sums(fetch_text(LICHESS_STANDARD_SHA256))
    expected = sha256_map.get(filename)
    url = LICHESS_STANDARD_INDEX + filename
    out_path = os.path.join(args.out_dir, filename)
    download_file(url, out_path, expected_sha256=expected, force=args.force)

    if args.decompress:
        out_pgn = out_path[:-4]
        decompress_zst(out_path, out_pgn, force=args.force)
        if not args.keep_zst:
            os.remove(out_path)

    print(f"Selected Lichess file: {filename}")
    return 0


def cmd_lichess_sample(args: argparse.Namespace) -> int:
    index_html = fetch_text(LICHESS_STANDARD_INDEX)
    entries = parse_lichess_standard_index(index_html)
    if not entries:
        raise RuntimeError("No entries found in Lichess standard index.")

    if args.month:
        filename = f"lichess_db_standard_rated_{args.month}.pgn.zst"
    else:
        filename = latest_lichess_standard_filename(index_html)
        if filename is None:
            raise RuntimeError("Failed to determine latest Lichess standard PGN.")

    url = LICHESS_STANDARD_INDEX + filename
    os.makedirs(os.path.dirname(args.out_pgn), exist_ok=True)
    out_pgn_file = open(args.out_pgn, "w", encoding="utf-8")
    out_fens_file = open(args.out_fens, "w", encoding="utf-8") if args.out_fens else None

    require_rated = None
    if args.rated and args.casual:
        raise SystemExit("Choose only one of --rated or --casual")
    if args.rated:
        require_rated = True
    if args.casual:
        require_rated = False

    time_class = None
    if args.time_class:
        allowed = {"ultrabullet", "bullet", "blitz", "rapid", "classical"}
        merged: list[str] = []
        for item in args.time_class:
            if item is None:
                continue
            merged.extend([part.strip() for part in item.split(",") if part.strip()])
        invalid = [item for item in merged if item not in allowed]
        if invalid:
            raise SystemExit(f"Invalid time-class entries: {invalid}")
        time_class = merged

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "lc0jax-downloader/1.0"})
        with urllib.request.urlopen(req) as resp:
            try:
                import zstandard as zstd  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("zstandard is required for streaming .zst files.") from exc

            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(resp) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
                kept = filter_pgn_stream(
                    text_stream,
                    out_pgn_file=out_pgn_file,
                    out_fens_file=out_fens_file,
                    max_games=args.max_games,
                    ply_stride=args.ply_stride,
                    min_elo=args.min_elo,
                    time_class=time_class,
                    require_rated=require_rated,
                    require_standard=True,
                    progress_every=args.progress_every,
                    progress_label="Sampled",
                )
    finally:
        out_pgn_file.close()
        if out_fens_file:
            out_fens_file.close()

    print(f"Selected Lichess file: {filename}")
    print(f"Sampled games kept: {kept}")
    return 0


def cmd_lc0_chunks(args: argparse.Namespace) -> int:
    index_html = fetch_text(LC0_TRAINING_INDEX)
    entries = parse_training_index(index_html)
    if not entries:
        raise RuntimeError("No LC0 training tar entries found.")

    picks = pick_latest_training_tars(
        entries,
        count=args.count,
        min_size=args.min_size,
        run=args.run,
    )
    if not picks:
        raise RuntimeError("No training tars matched the filter.")

    for entry in picks:
        url = LC0_TRAINING_INDEX + entry.filename
        out_path = os.path.join(args.out_dir, entry.filename)
        if args.dry_run:
            print(f"Would download {entry.filename} ({entry.size} bytes)")
            continue
        download_file(url, out_path, force=args.force)
        if args.extract:
            extract_tar(out_path, args.out_dir)

    return 0


def _open_puzzle_stream(path: str):
    if path.endswith(".zst"):
        try:
            import zstandard as zstd  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("zstandard is required to read puzzle .zst files.") from exc
        src = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(src)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
        return text_stream, src
    handle = open(path, "r", encoding="utf-8", errors="ignore")
    return handle, None


def cmd_lichess_puzzles(args: argparse.Namespace) -> int:
    if args.csv:
        puzzle_path = args.csv
        if not os.path.exists(puzzle_path):
            raise RuntimeError(f"Puzzle CSV not found: {puzzle_path}")
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        filename = os.path.basename(LICHESS_PUZZLE_URL)
        puzzle_path = os.path.join(args.out_dir, filename)
        download_file(LICHESS_PUZZLE_URL, puzzle_path, force=args.force)

    out_fens = args.out_fens
    if not out_fens:
        raise RuntimeError("--out-fens is required for lichess-puzzles.")
    os.makedirs(os.path.dirname(out_fens), exist_ok=True)

    try:
        import chess  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("python-chess is required to parse puzzle FENs.") from exc

    stream, src = _open_puzzle_stream(puzzle_path)
    kept = 0
    seen = 0
    try:
        reader = csv.DictReader(stream)
        with open(out_fens, "w", encoding="utf-8") as out_f:
            for row in reader:
                seen += 1
                try:
                    rating = int(row.get("Rating", "0"))
                except ValueError:
                    continue
                if rating < args.min_rating:
                    continue
                fen = row.get("FEN") or row.get("Fen") or row.get("fen")
                if not fen:
                    continue
                moves = row.get("Moves", "").strip()
                if not args.raw_fen:
                    if not moves:
                        continue
                    first_move = moves.split()[0]
                    try:
                        board = chess.Board(fen)
                        board.push_uci(first_move)
                        fen = board.fen()
                    except Exception:
                        continue
                out_f.write(fen + "\n")
                kept += 1
                if args.progress_every and kept % args.progress_every == 0:
                    print(f"Kept: {kept} (seen {seen})", flush=True)
                if args.max_positions is not None and kept >= args.max_positions:
                    break
    finally:
        stream.close()
        if src:
            src.close()

    print(f"Puzzle positions kept: {kept} (min_rating={args.min_rating})")
    return 0


def _iter_ndjson(text_stream):
    for line in text_stream:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _extract_broadcast_id(obj: dict) -> str | None:
    for key in ("id", "_id", "broadcastId", "tourId"):
        if key in obj:
            return obj[key]
    for parent_key in ("tour", "broadcast"):
        if parent_key in obj and isinstance(obj[parent_key], dict):
            for key in ("id", "_id"):
                if key in obj[parent_key]:
                    return obj[parent_key][key]
    return None


def _extract_round_ids(obj: dict) -> list[str]:
    rounds = obj.get("rounds")
    if not isinstance(rounds, list):
        return []
    ids = []
    for rnd in rounds:
        if isinstance(rnd, dict):
            rid = rnd.get("id") or rnd.get("_id")
            if rid:
                ids.append(rid)
    return ids


def cmd_lichess_broadcasts(args: argparse.Namespace) -> int:
    broadcast_ids: list[str] = []
    round_ids: list[str] = []

    if args.broadcast_id:
        broadcast_ids.extend(args.broadcast_id)
    if args.round_id:
        round_ids.extend(args.round_id)

    if not broadcast_ids and not round_ids:
        url = LICHESS_BROADCAST_API
        if args.nb is not None:
            url = f"{url}?nb={args.nb}"
        req = urllib.request.Request(url, headers={"User-Agent": "lc0jax-downloader/1.0"})
        with urllib.request.urlopen(req) as resp:
            text_stream = io.TextIOWrapper(resp, encoding="utf-8", errors="ignore")
            for obj in _iter_ndjson(text_stream):
                bid = _extract_broadcast_id(obj)
                if bid:
                    broadcast_ids.append(bid)
                if args.include_rounds:
                    round_ids.extend(_extract_round_ids(obj))
                if args.max_broadcasts and len(broadcast_ids) >= args.max_broadcasts:
                    break

    if not broadcast_ids and not round_ids:
        raise RuntimeError("No broadcast ids found. Provide --broadcast-id or --round-id.")

    out_pgn_path = args.out_pgn
    out_fens_path = args.out_fens
    os.makedirs(os.path.dirname(out_pgn_path), exist_ok=True)
    out_pgn_file = open(out_pgn_path, "w", encoding="utf-8")
    out_fens_file = open(out_fens_path, "w", encoding="utf-8") if out_fens_path else None

    time_class = None
    if args.time_class:
        allowed = {"ultrabullet", "bullet", "blitz", "rapid", "classical"}
        merged: list[str] = []
        for item in args.time_class:
            if item is None:
                continue
            merged.extend([part.strip() for part in item.split(",") if part.strip()])
        invalid = [item for item in merged if item not in allowed]
        if invalid:
            raise SystemExit(f"Invalid time-class entries: {invalid}")
        time_class = merged

    remaining = args.max_games
    sleep_seconds = max(args.sleep, 0.0)
    try:
        for bid in broadcast_ids:
            if remaining is not None and remaining <= 0:
                break
            url = f"{LICHESS_BROADCAST_API}/{bid}.pgn"
            print(f"Downloading broadcast {bid} -> {url}")
            attempts = 0
            while True:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "lc0jax-downloader/1.0"})
                    with urllib.request.urlopen(req) as resp:
                        text_stream = io.TextIOWrapper(resp, encoding="utf-8", errors="ignore")
                        kept = filter_pgn_stream(
                            text_stream,
                            out_pgn_file=out_pgn_file,
                            out_fens_file=out_fens_file,
                            max_games=remaining,
                            ply_stride=args.ply_stride,
                            min_elo=args.min_elo,
                            time_class=time_class,
                            require_rated=None,
                            require_standard=True,
                            progress_every=args.progress_every,
                            progress_label="Broadcast kept",
                        )
                        if remaining is not None:
                            remaining -= kept
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 429 and attempts < args.max_retries:
                        backoff = args.retry_backoff * (2 ** attempts)
                        print(f"HTTP 429 for {bid}; sleeping {backoff}s before retry {attempts + 1}.")
                        time.sleep(backoff)
                        attempts += 1
                        continue
                    print(f"Broadcast {bid} failed: {exc}")
                    break
            if sleep_seconds:
                time.sleep(sleep_seconds)

        for rid in round_ids:
            if remaining is not None and remaining <= 0:
                break
            url = f"{LICHESS_BROADCAST_API}/round/{rid}.pgn"
            print(f"Downloading broadcast round {rid} -> {url}")
            attempts = 0
            while True:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "lc0jax-downloader/1.0"})
                    with urllib.request.urlopen(req) as resp:
                        text_stream = io.TextIOWrapper(resp, encoding="utf-8", errors="ignore")
                        kept = filter_pgn_stream(
                            text_stream,
                            out_pgn_file=out_pgn_file,
                            out_fens_file=out_fens_file,
                            max_games=remaining,
                            ply_stride=args.ply_stride,
                            min_elo=args.min_elo,
                            time_class=time_class,
                            require_rated=None,
                            require_standard=True,
                            progress_every=args.progress_every,
                            progress_label="Broadcast kept",
                        )
                        if remaining is not None:
                            remaining -= kept
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 429 and attempts < args.max_retries:
                        backoff = args.retry_backoff * (2 ** attempts)
                        print(f"HTTP 429 for round {rid}; sleeping {backoff}s before retry {attempts + 1}.")
                        time.sleep(backoff)
                        attempts += 1
                        continue
                    print(f"Broadcast round {rid} failed: {exc}")
                    break
            if sleep_seconds:
                time.sleep(sleep_seconds)
    finally:
        out_pgn_file.close()
        if out_fens_file:
            out_fens_file.close()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    lichess = subparsers.add_parser("lichess", help="Download Lichess standard rated PGNs")
    lichess.add_argument("--out-dir", default="data/lichess")
    lichess.add_argument("--month", default=None, help="Explicit YYYY-MM month (e.g. 2025-12)")
    lichess.add_argument("--decompress", action="store_true")
    lichess.add_argument("--keep-zst", action="store_true")
    lichess.add_argument("--force", action="store_true")
    lichess.set_defaults(func=cmd_lichess)

    lichess_sample = subparsers.add_parser(
        "lichess-sample",
        help="Stream a Lichess standard dump and keep a filtered sample",
    )
    lichess_sample.add_argument("--out-pgn", default="data/lichess/lichess_sample.pgn")
    lichess_sample.add_argument("--out-fens", default=None)
    lichess_sample.add_argument("--month", default=None, help="Explicit YYYY-MM month (e.g. 2025-12)")
    lichess_sample.add_argument("--max-games", type=int, default=1000)
    lichess_sample.add_argument("--ply-stride", type=int, default=1)
    lichess_sample.add_argument("--min-elo", type=int, default=None)
    lichess_sample.add_argument(
        "--time-class",
        action="append",
        default=None,
        help="Time control class (repeatable or comma-separated)",
    )
    lichess_sample.add_argument("--rated", action="store_true")
    lichess_sample.add_argument("--casual", action="store_true")
    lichess_sample.add_argument("--progress-every", type=int, default=100)
    lichess_sample.set_defaults(func=cmd_lichess_sample)

    chunks = subparsers.add_parser("lc0-chunks", help="Download LC0 self-play training tarballs")
    chunks.add_argument("--out-dir", default="data/lc0-training")
    chunks.add_argument("--run", type=int, default=None)
    chunks.add_argument("--count", type=int, default=1)
    chunks.add_argument("--min-size", type=int, default=1_000_000)
    chunks.add_argument("--extract", action="store_true")
    chunks.add_argument("--dry-run", action="store_true")
    chunks.add_argument("--force", action="store_true")
    chunks.set_defaults(func=cmd_lc0_chunks)

    puzzles = subparsers.add_parser("lichess-puzzles", help="Download Lichess puzzles and extract FENs")
    puzzles.add_argument("--out-dir", default="data/lichess")
    puzzles.add_argument("--csv", default=None, help="Optional local puzzle CSV or .zst path")
    puzzles.add_argument("--out-fens", default="data/lichess/puzzles_2500.fens")
    puzzles.add_argument("--min-rating", type=int, default=2500)
    puzzles.add_argument("--max-positions", type=int, default=None)
    puzzles.add_argument("--raw-fen", action="store_true", help="Keep the raw puzzle FEN (before first move)")
    puzzles.add_argument("--progress-every", type=int, default=50000)
    puzzles.add_argument("--force", action="store_true")
    puzzles.set_defaults(func=cmd_lichess_puzzles)

    broadcasts = subparsers.add_parser(
        "lichess-broadcasts",
        help="Download Lichess broadcast PGNs via the broadcast API",
    )
    broadcasts.add_argument("--out-pgn", default="data/lichess/lichess_broadcasts.pgn")
    broadcasts.add_argument("--out-fens", default=None)
    broadcasts.add_argument("--broadcast-id", action="append", default=None)
    broadcasts.add_argument("--round-id", action="append", default=None)
    broadcasts.add_argument("--nb", type=int, default=None, help="Number of broadcasts to request from /api/broadcast")
    broadcasts.add_argument("--max-broadcasts", type=int, default=None)
    broadcasts.add_argument("--include-rounds", action="store_true")
    broadcasts.add_argument("--max-games", type=int, default=None)
    broadcasts.add_argument("--ply-stride", type=int, default=1)
    broadcasts.add_argument("--min-elo", type=int, default=None)
    broadcasts.add_argument(
        "--time-class",
        action="append",
        default=None,
        help="Time control class (repeatable or comma-separated)",
    )
    broadcasts.add_argument("--progress-every", type=int, default=100)
    broadcasts.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between broadcasts.")
    broadcasts.add_argument("--max-retries", type=int, default=5)
    broadcasts.add_argument("--retry-backoff", type=float, default=5.0)
    broadcasts.set_defaults(func=cmd_lichess_broadcasts)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
