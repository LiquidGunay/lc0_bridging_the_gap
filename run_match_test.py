import subprocess
import json

def run():
    cmd = [
        ".venv/bin/python",
        "tools/match_concepts.py",
        "--concepts", "data/runs/legacy_2026-02-02/concepts/svm_cvxpy/concept_direction.npz",
        "--activations", "/tmp/puzzles_activations",
        "--tags", "/tmp/puzzles.jsonl",
        "--out", "/tmp/match_report.json"
    ]
    subprocess.run(cmd, check=True)
    with open("/tmp/match_report.json") as f:
        print(json.dumps(json.load(f), indent=2))

if __name__ == "__main__":
    run()
