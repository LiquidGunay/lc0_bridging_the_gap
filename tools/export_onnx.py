"""Export LC0 .pb.gz to ONNX using the official lc0 leela2onnx command."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb", required=True, help="Path to .pb.gz network file")
    parser.add_argument("--onnx", required=True, help="Output ONNX path")
    parser.add_argument(
        "--lc0",
        default="lc0",
        help="Path to lc0 binary (will invoke `lc0 leela2onnx`)",
    )
    parser.add_argument(
        "--leela2onnx",
        default=None,
        help="Optional direct path to leela2onnx binary (overrides --lc0)",
    )
    args = parser.parse_args()

    pb_path = Path(args.pb)
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if args.leela2onnx:
        exe = args.leela2onnx
        cmd = [exe, f"--input={pb_path}", f"--output={onnx_path}"]
    else:
        exe = args.lc0
        cmd = [exe, "leela2onnx", f"--input={pb_path}", f"--output={onnx_path}"]

    if shutil.which(exe) is None and not Path(exe).exists():
        raise FileNotFoundError(
            f"Executable not found: {exe}. Provide --lc0 or --leela2onnx."
        )

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(
            "leela2onnx failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )

    if not onnx_path.exists():
        raise RuntimeError(f"ONNX export succeeded but file missing: {onnx_path}")

    print(f"Wrote ONNX: {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
