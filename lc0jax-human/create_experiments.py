import json

experiments = []

# Version suffix for fresh metadata and clean start
v = "v21-dfm"

# Mega Scale-Up Run (1 Million Steps / ~250 Epochs)
prioritized = [
    {"layers": 4, "h": 8, "type": "modern"},
    {"layers": 12, "h": 8, "type": "modern"},
    {"layers": 12, "h": 8, "type": "baseline"},
]

for cfg in prioritized:
    layers = cfg["layers"]
    h = cfg["h"]
    is_modern = (cfg["type"] == "modern")

    exp_id = f"dfm-{cfg['type']}-l{layers}-h{h}-{v}"

    flags = "--use-qk-gain --use-muon" if is_modern else ""

    experiments.append({
        "experiment_id": exp_id,
        "entry_command": (
            f"/tmp/venv/bin/python -u scripts/train_dfm.py "
            f"--steps 1000000 --batch-size 256 --token-dim 512 "
            f"--num-layers {layers} --num-heads 8 --mlp-dim 2048 "
            f"{flags} "
            f"--chunk-dir /tmp/lc0jaxhuman/chunks "
            f"--run-id {exp_id} --resume "
            f"--save-dir /tmp/lc0jaxhuman/artifacts "
            f"--checkpoint-uri gs://{{REGION_BUCKET}}/runs/dfm/{exp_id}/checkpoints "
            f"--save-every 1000 --horizon {h}"
        )
    })

with open("experiments.jsonl", "w") as f:
    for exp in experiments:
        f.write(json.dumps(exp) + "\n")
