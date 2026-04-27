#!/bin/bash
set -e

cd ~/schutpaper
source .venv/bin/activate

BASE_DIR="data/runs/test_eval_pipeline"
mkdir -p $BASE_DIR/{lichess,lc0-training,activations,concepts,logs}

HUMAN_FENS="/tmp/human.fens"
LC0_FENS="/tmp/lc0.fens"

# 1. Eval filter the human fens
EVAL_OUT_FENS="${BASE_DIR}/lichess/human.eval.fens"
echo "Running eval filter on human FENs..."
python tools/filter_fens_eval.py \
  --fens $HUMAN_FENS \
  --out $EVAL_OUT_FENS \
  --lc0 /tmp/lc0-src/build/release/lc0 \
  --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz \
  --nodes 100 \
  --progress-every 10

# 2. Dump activations for human
HUMAN_ACT_OUT="${BASE_DIR}/activations/human"
echo "Dumping human activations..."
python tools/dump_activations.py \
  --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz \
  --fens $EVAL_OUT_FENS \
  --out $HUMAN_ACT_OUT \
  --batch-size 64

# 3. Dump activations for LC0
LC0_ACT_OUT="${BASE_DIR}/activations/lc0"
echo "Dumping LC0 activations..."
python tools/dump_activations.py \
  --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz \
  --fens $LC0_FENS \
  --out $LC0_ACT_OUT \
  --batch-size 64

# 4. Discover concepts using the new svm_cvxpy audited method
CONCEPT_OUT="${BASE_DIR}/concepts/svm_cvxpy"
echo "Discovering concepts (svm_cvxpy)..."
python tools/discover_concepts.py \
  --embeddings-a $LC0_ACT_OUT \
  --embeddings-b $HUMAN_ACT_OUT \
  --out $CONCEPT_OUT \
  --method svm_cvxpy \
  --patch --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz

echo "Done!"
cat ${CONCEPT_OUT}/report.json
