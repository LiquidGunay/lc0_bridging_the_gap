#!/bin/bash
set -e

# Temporary test directory
export BASE_DIR="data/runs/test_pipeline"
mkdir -p $BASE_DIR

# Keep it small for testing
export BROADCAST_GAMES=200
export LC0_MAX_POSITIONS=200
export SHARD_SIZE=100

# Enable our new filters
export EVAL_FILTER=1
export EVAL_NODES=100
export EVAL_MIN_CP=-150
export EVAL_MAX_CP=150

export FILTER_HUMAN=1
export FILTER_LC0=1
export FILTER_MIN_PLY=20

# Run the pipeline
./tools/run_full_pipeline.sh
