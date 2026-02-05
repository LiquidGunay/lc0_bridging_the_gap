"""Helpers for parsing LC0 training data index pages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re


_TRAINING_TAR_RE = re.compile(r"training-run(\d+)--(\d{8})-(\d{4,6})\.tar(?:\.gz)?")


@dataclass(frozen=True)
class TrainingTarEntry:
    filename: str
    size: int
    run: int
    timestamp: datetime


def parse_training_filename(filename: str) -> tuple[int, datetime]:
    """Parse a training tar filename into (run_id, timestamp)."""
    match = _TRAINING_TAR_RE.fullmatch(filename)
    if not match:
        raise ValueError(f"Unrecognized training tar filename: {filename}")
    run_id = int(match.group(1))
    date_str = match.group(2)
    time_str = match.group(3)
    if len(time_str) == 4:
        fmt = "%Y%m%d%H%M"
    else:
        fmt = "%Y%m%d%H%M%S"
    timestamp = datetime.strptime(f"{date_str}{time_str}", fmt)
    return run_id, timestamp


def parse_training_index(html: str) -> list[TrainingTarEntry]:
    """Parse the LC0 training data index HTML into tar entries."""
    entries: list[TrainingTarEntry] = []
    for line in html.splitlines():
        if "training-run" not in line:
            continue
        filename_match = re.search(
            r'href="(training-run\d+--\d{8}-\d{4,6}\.tar(?:\.gz)?)"',
            line,
        )
        if not filename_match:
            continue
        filename = filename_match.group(1)
        size_match = re.search(r"\s(\d+)\s*$", line)
        size = int(size_match.group(1)) if size_match else 0
        run_id, timestamp = parse_training_filename(filename)
        entries.append(TrainingTarEntry(filename=filename, size=size, run=run_id, timestamp=timestamp))
    return entries


def pick_latest_training_tars(
    entries: list[TrainingTarEntry],
    *,
    count: int = 1,
    min_size: int = 0,
    run: int | None = None,
) -> list[TrainingTarEntry]:
    """Return up to `count` newest tar entries filtered by size/run."""
    filtered = [
        entry
        for entry in entries
        if entry.size >= min_size and (run is None or entry.run == run)
    ]
    filtered.sort(key=lambda entry: (entry.timestamp, entry.size), reverse=True)
    return filtered[:count]


def latest_training_tar(
    entries: list[TrainingTarEntry],
    *,
    min_size: int = 0,
    run: int | None = None,
) -> TrainingTarEntry | None:
    """Return the newest tar entry by timestamp, or None if none match."""
    picks = pick_latest_training_tars(entries, count=1, min_size=min_size, run=run)
    return picks[0] if picks else None
