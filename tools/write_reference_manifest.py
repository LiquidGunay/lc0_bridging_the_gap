"""Write human/machine reference dataset manifests."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from lc0jax.interpretability.manifests import file_manifest, reference_dataset_manifest


def _coerce_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_key_value(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE item, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in item: {item}")
        parsed[key] = _coerce_value(value.strip())
    return parsed


def _validate_existing_files(paths: list[str]) -> None:
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Reference manifest input/output paths do not exist: "
            + ", ".join(missing)
            + ". Use --allow-missing to write a planned manifest."
        )


def _file_records(
    paths: list[str],
    *,
    role: str,
    checksum: bool,
    count_lines: bool,
) -> list[dict[str, Any]]:
    return [
        file_manifest(path, role=role, checksum=checksum, count_lines=count_lines)
        for path in paths
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=["human", "machine"],
        required=True,
        help="Reference manifest family to write.",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Local source/input file path. Repeat for multiple files.",
    )
    parser.add_argument(
        "--output",
        action="append",
        default=[],
        help="Local derived output path, such as filtered PGN/FEN/records.",
    )
    parser.add_argument(
        "--source-url",
        action="append",
        default=[],
        help="Public or private source URL for later lookup.",
    )
    parser.add_argument("--source-type", default="mixed")
    parser.add_argument("--format", default="pgn")
    parser.add_argument("--min-elo", type=int, default=None)
    parser.add_argument("--max-elo", type=int, default=None)
    parser.add_argument("--time-class", action="append", default=[])
    parser.add_argument("--rated", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-ply", type=int, default=None)
    parser.add_argument("--max-ply", type=int, default=None)
    parser.add_argument("--min-phase", type=float, default=None)
    parser.add_argument("--max-phase", type=float, default=None)
    parser.add_argument("--min-pieces", type=int, default=None)
    parser.add_argument("--min-nonpawn", type=int, default=None)
    parser.add_argument("--dedupe-key", default="board_fen side castling ep")
    parser.add_argument("--split-key", default="game_id")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument(
        "--count",
        action="append",
        default=[],
        help="Additional count metadata as KEY=VALUE. Repeatable.",
    )
    parser.add_argument("--note", default=None)
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Do not hash local files. Useful for huge staged artifacts.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow local input/output paths that do not exist yet.",
    )
    parser.add_argument(
        "--count-lines",
        action="store_true",
        help="Count non-empty lines in local files. Can be slow for large PGNs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    kind = f"{args.kind}_reference_v1"
    checksum = not args.no_checksum
    if not args.allow_missing:
        _validate_existing_files([*args.input, *args.output])
    filters = {
        "format": args.format,
        "min_elo": args.min_elo,
        "max_elo": args.max_elo,
        "time_classes": args.time_class,
        "rated": args.rated,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "min_phase": args.min_phase,
        "max_phase": args.max_phase,
        "min_pieces": args.min_pieces,
        "min_nonpawn": args.min_nonpawn,
    }
    filters = {key: value for key, value in filters.items() if value not in (None, [])}
    manifest = reference_dataset_manifest(
        kind=kind,
        created_utc=_dt.datetime.now(tz=_dt.UTC).isoformat(),
        name=args.name,
        source={
            "type": args.source_type,
            "urls": args.source_url,
        },
        inputs=_file_records(
            args.input,
            role="input",
            checksum=checksum,
            count_lines=args.count_lines,
        ),
        outputs=_file_records(
            args.output,
            role="output",
            checksum=checksum,
            count_lines=args.count_lines,
        ),
        filters=filters,
        dedupe={"key": args.dedupe_key},
        split={"key": args.split_key},
        exclusions=args.exclude,
        counts=_parse_key_value(args.count),
        notes=args.note,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {kind} manifest to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
