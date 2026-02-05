"""Run LC0 benchmark modes with a specific weights file."""

from __future__ import annotations

import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc0", required=True, help="Path to lc0 binary")
    parser.add_argument("--weights", required=True, help="Path to .pb.gz weights")
    parser.add_argument(
        "--mode",
        default="benchmark",
        choices=["benchmark", "bench", "backendbench"],
        help="LC0 benchmark mode",
    )
    parser.add_argument("--backend", default=None, help="LC0 backend (eigen, cudnn, etc.)")
    parser.add_argument("--backend-opts", default=None, help="LC0 backend options string")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--extra", nargs="*", default=[], help="Extra args passed to lc0")
    args = parser.parse_args()

    cmd = [args.lc0, args.mode, f"--weights={args.weights}"]
    if args.backend:
        cmd.append(f"--backend={args.backend}")
    if args.backend_opts:
        cmd.append(f"--backend-opts={args.backend_opts}")
    if args.threads is not None:
        cmd.append(f"--threads={args.threads}")
    if args.minibatch_size is not None:
        cmd.append(f"--minibatch-size={args.minibatch_size}")
    cmd.extend(args.extra)

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
