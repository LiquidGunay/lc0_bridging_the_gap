"""Download the target BT4 LC0 model and print its SHA256."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://storage.lczero.org/files/networks-contrib/"
    "BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "lc0jax/0.1"})
    with urllib.request.urlopen(req) as resp, out_path.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz")
    args = parser.parse_args()

    out_path = Path(args.out)
    download(args.url, out_path)
    digest = sha256sum(out_path)
    size = out_path.stat().st_size
    print(f"Downloaded {out_path} ({size} bytes)")
    print(f"SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
