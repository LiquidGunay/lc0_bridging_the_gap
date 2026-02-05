"""Generate Python protobuf bindings from vendored LC0 .proto files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protoc",
        default="protoc",
        help="Path to protoc compiler",
    )
    parser.add_argument(
        "--out",
        default="lc0jax",
        help="Output root for *_pb2.py files (will create proto/ under it)",
    )
    args = parser.parse_args()

    protoc = args.protoc
    use_grpc_tools = False
    if shutil.which(protoc) is None and not Path(protoc).exists():
        use_grpc_tools = True

    proto_root = Path("lc0jax")
    proto_dir = proto_root / "proto"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    protos = [str(proto_dir / "net.proto"), str(proto_dir / "chunk.proto")]
    if use_grpc_tools:
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto_root}",
            f"--python_out={out_dir}",
        ] + protos
    else:
        cmd = [protoc, f"-I{proto_root}", f"--python_out={out_dir}"] + protos
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "protoc failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    print("Generated protobuf bindings in", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
