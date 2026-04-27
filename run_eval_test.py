import subprocess
import os

def run():
    # Let's just run the new filter eval tool directly on existing legacy FENs
    # to prove it works end-to-end without downloading all the data.
    cmd = [
        ".venv/bin/python",
        "tools/filter_fens_eval.py",
        "--fens", "data/runs/legacy_2026-02-02/lichess/lichess_sample_2000_rapid_classical_5k.fens",
        "--out", "/tmp/eval_filtered.fens",
        "--lc0", "/tmp/lc0-src/build/release/lc0",
        "--weights", "models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz",
        "--nodes", "100",
        "--min-cp", "-150",
        "--max-cp", "150",
        "--min-ply", "20",
        "--max-positions", "10",
        "--progress-every", "1"
    ]
    subprocess.run(cmd, check=True)

    with open("/tmp/eval_filtered.fens") as f:
        print(f.read())

if __name__ == "__main__":
    run()
