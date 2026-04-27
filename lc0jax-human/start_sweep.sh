#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Waiting for chunk data in GCS..."
while true; do
    if /snap/google-cloud-cli/current/bin/gcloud storage ls "gs://gunay-chess-experiments-us-central2/data/chunks/**" | grep -qE '\.(zst|gz|npz)$'; then
        echo "Data found! Proceeding with sweep."
        break
    fi
    echo "No chunk data found. Waiting 5 minutes..."
    sleep 300
done

.venv311/bin/python -u scripts/sweep_manager.py --experiments experiments.jsonl 2>&1 | tee sweep.log
