#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing python at $PYTHON_BIN (set PYTHON_BIN to override)." >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +\"%Y%m%d_%H%M%S\")}"
BASE_DIR="${BASE_DIR:-data/runs/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

mkdir -p "$LOG_DIR" "$BASE_DIR"/{lichess,lc0-training,activations,concepts}

# Broadcast (human) dataset parameters.
BROADCAST_GAMES="${BROADCAST_GAMES:-10000}"
BROADCAST_MIN_ELO="${BROADCAST_MIN_ELO:-2400}"
BROADCAST_TIME_CLASS="${BROADCAST_TIME_CLASS:-classical}"
BROADCAST_NB="${BROADCAST_NB:-1000}"
BROADCAST_MAX_BROADCASTS="${BROADCAST_MAX_BROADCASTS:-1000}"
BROADCAST_SLEEP="${BROADCAST_SLEEP:-0.5}"
BROADCAST_MAX_RETRIES="${BROADCAST_MAX_RETRIES:-5}"
BROADCAST_RETRY_BACKOFF="${BROADCAST_RETRY_BACKOFF:-5.0}"
BROADCAST_OUT_PGN="${BROADCAST_OUT_PGN:-${BASE_DIR}/lichess/broadcasts_2400_classical.pgn}"
BROADCAST_OUT_FENS="${BROADCAST_OUT_FENS:-${BASE_DIR}/lichess/broadcasts_2400_classical.fens}"

# LC0 self-play parameters.
LC0_CHUNK_COUNT="${LC0_CHUNK_COUNT:-2}"
LC0_MAX_POSITIONS="${LC0_MAX_POSITIONS:-100000}"
LC0_CHUNK_DIR="${LC0_CHUNK_DIR:-${BASE_DIR}/lc0-training}"
LC0_OUT_PGN="${LC0_OUT_PGN:-${BASE_DIR}/lc0-training/lc0_100k.pgn}"
LC0_OUT_FENS="${LC0_OUT_FENS:-${BASE_DIR}/lc0-training/lc0_100k.fens}"

# Activation dump parameters.
PB_FILE="${PB_FILE:-models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SHARD_SIZE="${SHARD_SIZE:-4096}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5000}"

HUMAN_ACT_OUT="${HUMAN_ACT_OUT:-${BASE_DIR}/activations/human_broadcasts_2400_classical}"
LC0_ACT_OUT="${LC0_ACT_OUT:-${BASE_DIR}/activations/lc0_100k}"

# Optional FEN filtering (disabled by default).
FILTER_HUMAN="${FILTER_HUMAN:-0}"
FILTER_LC0="${FILTER_LC0:-0}"
FILTER_MIN_PLY="${FILTER_MIN_PLY:-}"
FILTER_MAX_PLY="${FILTER_MAX_PLY:-}"
FILTER_MIN_PHASE="${FILTER_MIN_PHASE:-}"
FILTER_MAX_PHASE="${FILTER_MAX_PHASE:-}"
FILTER_MIN_PIECES="${FILTER_MIN_PIECES:-}"
FILTER_MAX_PIECES="${FILTER_MAX_PIECES:-}"
FILTER_MIN_NONPAWN="${FILTER_MIN_NONPAWN:-}"
FILTER_MAX_NONPAWN="${FILTER_MAX_NONPAWN:-}"
FILTER_DEDUPE="${FILTER_DEDUPE:-0}"
FILTER_PROGRESS_EVERY="${FILTER_PROGRESS_EVERY:-}"
FILTER_HUMAN_OUT_FENS="${FILTER_HUMAN_OUT_FENS:-${BASE_DIR}/lichess/broadcasts_2400_classical.filtered.fens}"
FILTER_LC0_OUT_FENS="${FILTER_LC0_OUT_FENS:-${BASE_DIR}/lc0-training/lc0_100k.filtered.fens}"

# Optional policy-disagreement filtering (disabled by default).
DISAGREE_FILTER="${DISAGREE_FILTER:-0}"
DISAGREE_LC0_BIN="${DISAGREE_LC0_BIN:-/tmp/lc0-src/build/release/lc0}"
DISAGREE_WEIGHTS="${DISAGREE_WEIGHTS:-$PB_FILE}"
DISAGREE_ONNX="${DISAGREE_ONNX:-}"
DISAGREE_OUT_FENS="${DISAGREE_OUT_FENS:-${BASE_DIR}/lichess/broadcasts_2400_classical.disagree.fens}"
DISAGREE_NODES="${DISAGREE_NODES:-800}"
DISAGREE_MOVETIME_MS="${DISAGREE_MOVETIME_MS:-}"
DISAGREE_BATCH_SIZE="${DISAGREE_BATCH_SIZE:-64}"
DISAGREE_MAX_POSITIONS="${DISAGREE_MAX_POSITIONS:-}"
DISAGREE_PROGRESS_EVERY="${DISAGREE_PROGRESS_EVERY:-100}"
DISAGREE_START_LINE="${DISAGREE_START_LINE:-}"
DISAGREE_APPEND="${DISAGREE_APPEND:-0}"
DISAGREE_SHARD_INDEX="${DISAGREE_SHARD_INDEX:-0}"
DISAGREE_SHARD_COUNT="${DISAGREE_SHARD_COUNT:-1}"
DISAGREE_STATE_FILE="${DISAGREE_STATE_FILE:-}"
DISAGREE_STATE_EVERY="${DISAGREE_STATE_EVERY:-}"
DISAGREE_THREADS="${DISAGREE_THREADS:-}"
DISAGREE_BACKEND="${DISAGREE_BACKEND:-}"
DISAGREE_BACKEND_OPTS="${DISAGREE_BACKEND_OPTS:-}"
DISAGREE_UCI_TIMEOUT="${DISAGREE_UCI_TIMEOUT:-60}"

# Concept parameters.
CONCEPT_MAX_SAMPLES="${CONCEPT_MAX_SAMPLES:-100000}"
CONCEPT_K="${CONCEPT_K:-8}"
CAUSAL_MAX_SAMPLES="${CAUSAL_MAX_SAMPLES:-2000}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

should_skip_file() {
  local path="$1"
  if [ "$SKIP_EXISTING" -eq 1 ] && [ -f "$path" ] && [ -s "$path" ]; then
    return 0
  fi
  return 1
}

should_skip_dir_done() {
  local dir="$1"
  if [ "$SKIP_EXISTING" -eq 1 ] && [ -f "$dir/done.txt" ]; then
    return 0
  fi
  return 1
}

resolve_chunk_dir() {
  local dir="$LC0_CHUNK_DIR"
  shopt -s nullglob
  local files=("$dir"/*.gz)
  if [ ${#files[@]} -gt 0 ]; then
    echo "$dir"
    return 0
  fi
  if [ "$SKIP_EXISTING" -eq 1 ]; then
    if [ -d "data/lc0-training" ]; then
      files=(data/lc0-training/*.gz)
      if [ ${#files[@]} -gt 0 ]; then
        echo "data/lc0-training"
        return 0
      fi
    fi
    local legacy_dirs
    legacy_dirs=$(ls -d data/runs/legacy_*/lc0-training 2>/dev/null || true)
    for legacy in $legacy_dirs; do
      files=("$legacy"/*.gz)
      if [ ${#files[@]} -gt 0 ]; then
        echo "$legacy"
        return 0
      fi
    done
  fi
  echo "$dir"
}

run_fg() {
  echo "[$(timestamp)] $*"
  "$@"
}

run_bg() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}.log"
  echo "[$(timestamp)] starting ${name} -> ${log}"
  "$@" >"$log" 2>&1 &
  LAST_PID=$!
}

count_lines() {
  if [ -f "$1" ]; then
    wc -l < "$1" | tr -d " "
  else
    echo 0
  fi
}

pids=()
if ! should_skip_file "$BROADCAST_OUT_FENS"; then
  run_bg "broadcasts_download" "$PYTHON_BIN" tools/download_data.py lichess-broadcasts \
    --out-pgn "$BROADCAST_OUT_PGN" \
    --out-fens "$BROADCAST_OUT_FENS" \
    --max-games "$BROADCAST_GAMES" \
    --min-elo "$BROADCAST_MIN_ELO" \
    --time-class "$BROADCAST_TIME_CLASS" \
    --include-rounds --nb "$BROADCAST_NB" --max-broadcasts "$BROADCAST_MAX_BROADCASTS" \
    --progress-every 100 \
    --sleep "$BROADCAST_SLEEP" \
    --max-retries "$BROADCAST_MAX_RETRIES" \
    --retry-backoff "$BROADCAST_RETRY_BACKOFF"
  pids+=("$LAST_PID")
else
  existing_count=$(count_lines "$BROADCAST_OUT_FENS")
  echo "[$(timestamp)] skipping broadcasts download (existing $BROADCAST_OUT_FENS, lines=$existing_count)"
  if [ "$existing_count" -gt "$BROADCAST_GAMES" ]; then
    echo "[$(timestamp)] note: existing broadcast FENs exceed target ($BROADCAST_GAMES). Set SKIP_EXISTING=0 or BROADCAST_OUT_FENS to a new path to cap."
  fi
fi

if [ "$SKIP_EXISTING" -eq 1 ] && ls "$LC0_CHUNK_DIR"/* >/dev/null 2>&1; then
  echo "[$(timestamp)] skipping LC0 chunk download (files present)"
else
  run_bg "lc0_chunks_download" "$PYTHON_BIN" tools/download_data.py lc0-chunks \
    --out-dir "$LC0_CHUNK_DIR" \
    --count "$LC0_CHUNK_COUNT" \
    --extract
  pids+=("$LAST_PID")
fi

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[$(timestamp)] broadcasts kept: $(count_lines "$BROADCAST_OUT_FENS") (target $BROADCAST_GAMES)"

# Convert LC0 chunks to FEN/PGN.
CHUNK_DIR="$(resolve_chunk_dir)"
chunk_list_file="$BASE_DIR/lc0-training/chunk_list.txt"
find "$CHUNK_DIR" -type f -name "*.gz" | sort > "$chunk_list_file"
if [ ! -s "$chunk_list_file" ]; then
  echo "No LC0 chunks found under $CHUNK_DIR. Set LC0_CHUNK_DIR to an existing chunk folder or re-run without SKIP_EXISTING." >&2
  exit 1
fi

if should_skip_file "$LC0_OUT_FENS"; then
  echo "[$(timestamp)] skipping chunk conversion (existing $LC0_OUT_FENS)"
else
  run_fg "$PYTHON_BIN" tools/chunks_to_pgn.py \
    --chunk-list "$chunk_list_file" \
    --out-pgn "$LC0_OUT_PGN" \
    --out-fens "$LC0_OUT_FENS" \
    --max-positions "$LC0_MAX_POSITIONS"
fi

FILTER_ARGS=()
if [ -n "$FILTER_MIN_PLY" ]; then FILTER_ARGS+=(--min-ply "$FILTER_MIN_PLY"); fi
if [ -n "$FILTER_MAX_PLY" ]; then FILTER_ARGS+=(--max-ply "$FILTER_MAX_PLY"); fi
if [ -n "$FILTER_MIN_PHASE" ]; then FILTER_ARGS+=(--min-phase "$FILTER_MIN_PHASE"); fi
if [ -n "$FILTER_MAX_PHASE" ]; then FILTER_ARGS+=(--max-phase "$FILTER_MAX_PHASE"); fi
if [ -n "$FILTER_MIN_PIECES" ]; then FILTER_ARGS+=(--min-pieces "$FILTER_MIN_PIECES"); fi
if [ -n "$FILTER_MAX_PIECES" ]; then FILTER_ARGS+=(--max-pieces "$FILTER_MAX_PIECES"); fi
if [ -n "$FILTER_MIN_NONPAWN" ]; then FILTER_ARGS+=(--min-nonpawn "$FILTER_MIN_NONPAWN"); fi
if [ -n "$FILTER_MAX_NONPAWN" ]; then FILTER_ARGS+=(--max-nonpawn "$FILTER_MAX_NONPAWN"); fi
if [ "$FILTER_DEDUPE" -eq 1 ]; then FILTER_ARGS+=(--dedupe); fi
if [ -n "$FILTER_PROGRESS_EVERY" ]; then FILTER_ARGS+=(--progress-every "$FILTER_PROGRESS_EVERY"); fi

HUMAN_FENS_FOR_ACT="$BROADCAST_OUT_FENS"
LC0_FENS_FOR_ACT="$LC0_OUT_FENS"

if [ "$FILTER_HUMAN" -eq 1 ]; then
  if should_skip_file "$FILTER_HUMAN_OUT_FENS"; then
    echo "[$(timestamp)] skipping human FEN filtering (existing $FILTER_HUMAN_OUT_FENS)"
  else
    run_fg "$PYTHON_BIN" tools/filter_fens.py \
      --fens "$BROADCAST_OUT_FENS" \
      --out "$FILTER_HUMAN_OUT_FENS" \
      "${FILTER_ARGS[@]}"
  fi
  HUMAN_FENS_FOR_ACT="$FILTER_HUMAN_OUT_FENS"
fi

if [ "$FILTER_LC0" -eq 1 ]; then
  if should_skip_file "$FILTER_LC0_OUT_FENS"; then
    echo "[$(timestamp)] skipping LC0 FEN filtering (existing $FILTER_LC0_OUT_FENS)"
  else
    run_fg "$PYTHON_BIN" tools/filter_fens.py \
      --fens "$LC0_OUT_FENS" \
      --out "$FILTER_LC0_OUT_FENS" \
      "${FILTER_ARGS[@]}"
  fi
  LC0_FENS_FOR_ACT="$FILTER_LC0_OUT_FENS"
fi

if [ "$DISAGREE_FILTER" -eq 1 ]; then
  if should_skip_file "$DISAGREE_OUT_FENS"; then
    echo "[$(timestamp)] skipping disagreement filter (existing $DISAGREE_OUT_FENS)"
  else
    if [ ! -x "$DISAGREE_LC0_BIN" ]; then
      echo "Missing LC0 binary at $DISAGREE_LC0_BIN (set DISAGREE_LC0_BIN)." >&2
      exit 1
    fi
    DISAGREE_ARGS=(--fens "$HUMAN_FENS_FOR_ACT" --out "$DISAGREE_OUT_FENS" --pb "$PB_FILE" --lc0 "$DISAGREE_LC0_BIN")
    DISAGREE_ARGS+=(--nodes "$DISAGREE_NODES" --batch-size "$DISAGREE_BATCH_SIZE" --progress-every "$DISAGREE_PROGRESS_EVERY")
    if [ -n "$DISAGREE_MOVETIME_MS" ]; then DISAGREE_ARGS+=(--movetime-ms "$DISAGREE_MOVETIME_MS"); fi
    if [ -n "$DISAGREE_MAX_POSITIONS" ]; then DISAGREE_ARGS+=(--max-positions "$DISAGREE_MAX_POSITIONS"); fi
    if [ -n "$DISAGREE_THREADS" ]; then DISAGREE_ARGS+=(--threads "$DISAGREE_THREADS"); fi
    if [ -n "$DISAGREE_BACKEND" ]; then DISAGREE_ARGS+=(--backend "$DISAGREE_BACKEND"); fi
    if [ -n "$DISAGREE_BACKEND_OPTS" ]; then DISAGREE_ARGS+=(--backend-opts "$DISAGREE_BACKEND_OPTS"); fi
    if [ -n "$DISAGREE_UCI_TIMEOUT" ]; then DISAGREE_ARGS+=(--uci-timeout "$DISAGREE_UCI_TIMEOUT"); fi
    if [ -n "$DISAGREE_WEIGHTS" ]; then DISAGREE_ARGS+=(--weights "$DISAGREE_WEIGHTS"); fi
    if [ -n "$DISAGREE_ONNX" ]; then DISAGREE_ARGS+=(--onnx "$DISAGREE_ONNX"); fi
    if [ -n "$DISAGREE_START_LINE" ]; then DISAGREE_ARGS+=(--start-line "$DISAGREE_START_LINE"); fi
    if [ "$DISAGREE_APPEND" -eq 1 ]; then DISAGREE_ARGS+=(--append); fi
    if [ -n "$DISAGREE_SHARD_COUNT" ]; then DISAGREE_ARGS+=(--shard-count "$DISAGREE_SHARD_COUNT"); fi
    if [ -n "$DISAGREE_SHARD_INDEX" ]; then DISAGREE_ARGS+=(--shard-index "$DISAGREE_SHARD_INDEX"); fi
    if [ -n "$DISAGREE_STATE_FILE" ]; then DISAGREE_ARGS+=(--state-file "$DISAGREE_STATE_FILE"); fi
    if [ -n "$DISAGREE_STATE_EVERY" ]; then DISAGREE_ARGS+=(--state-every "$DISAGREE_STATE_EVERY"); fi
    run_fg "$PYTHON_BIN" tools/filter_fens_disagreement.py "${DISAGREE_ARGS[@]}"
  fi
  HUMAN_FENS_FOR_ACT="$DISAGREE_OUT_FENS"
fi

# Activation dumps (sequential).
if should_skip_dir_done "$HUMAN_ACT_OUT"; then
  echo "[$(timestamp)] skipping human activation dump (done.txt present)"
else
  run_fg "$PYTHON_BIN" tools/dump_activations.py \
    --pb "$PB_FILE" \
    --fens "$HUMAN_FENS_FOR_ACT" \
    --out "$HUMAN_ACT_OUT" \
    --batch-size "$BATCH_SIZE" \
    --shard-size "$SHARD_SIZE" \
    --progress-every "$PROGRESS_EVERY" \
    --count-fens
fi

if should_skip_dir_done "$LC0_ACT_OUT"; then
  echo "[$(timestamp)] skipping LC0 activation dump (done.txt present)"
else
  run_fg "$PYTHON_BIN" tools/dump_activations.py \
    --pb "$PB_FILE" \
    --fens "$LC0_FENS_FOR_ACT" \
    --out "$LC0_ACT_OUT" \
    --batch-size "$BATCH_SIZE" \
    --shard-size "$SHARD_SIZE" \
    --progress-every "$PROGRESS_EVERY" \
    --count-fens
fi

# Concept discovery + causal validation.
if should_skip_file "$BASE_DIR/concepts/full_mean_diff/report.json"; then
  echo "[$(timestamp)] skipping mean_diff concepts"
else
  run_fg "$PYTHON_BIN" tools/discover_concepts.py \
    --embeddings-a "$LC0_ACT_OUT" \
    --embeddings-b "$HUMAN_ACT_OUT" \
    --out "$BASE_DIR/concepts/full_mean_diff" \
    --method mean_diff \
    --max-samples "$CONCEPT_MAX_SAMPLES" \
    --patch --pb "$PB_FILE"
fi

if should_skip_file "$BASE_DIR/concepts/full_mean_diff/causal_report.json"; then
  echo "[$(timestamp)] skipping mean_diff causal validation"
else
  run_fg "$PYTHON_BIN" tools/causal_validate.py \
    --concept "$BASE_DIR/concepts/full_mean_diff" \
    --embeddings "$HUMAN_ACT_OUT" \
    --pb "$PB_FILE" \
    --max-samples "$CAUSAL_MAX_SAMPLES"
fi

if should_skip_file "$BASE_DIR/concepts/full_whitened/report.json"; then
  echo "[$(timestamp)] skipping whitened_mean_diff concepts"
else
  run_fg "$PYTHON_BIN" tools/discover_concepts.py \
    --embeddings-a "$LC0_ACT_OUT" \
    --embeddings-b "$HUMAN_ACT_OUT" \
    --out "$BASE_DIR/concepts/full_whitened" \
    --method whitened_mean_diff \
    --max-samples "$CONCEPT_MAX_SAMPLES" \
    --patch --pb "$PB_FILE"
fi

if should_skip_file "$BASE_DIR/concepts/full_whitened/causal_report.json"; then
  echo "[$(timestamp)] skipping whitened_mean_diff causal validation"
else
  run_fg "$PYTHON_BIN" tools/causal_validate.py \
    --concept "$BASE_DIR/concepts/full_whitened" \
    --embeddings "$HUMAN_ACT_OUT" \
    --pb "$PB_FILE" \
    --max-samples "$CAUSAL_MAX_SAMPLES"
fi

if should_skip_file "$BASE_DIR/concepts/full_cov_shift/report.json"; then
  echo "[$(timestamp)] skipping cov_shift concepts"
else
  run_fg "$PYTHON_BIN" tools/discover_concepts.py \
    --embeddings-a "$LC0_ACT_OUT" \
    --embeddings-b "$HUMAN_ACT_OUT" \
    --out "$BASE_DIR/concepts/full_cov_shift" \
    --method cov_shift --k "$CONCEPT_K" \
    --max-samples "$CONCEPT_MAX_SAMPLES" \
    --patch --pb "$PB_FILE"
fi

if should_skip_file "$BASE_DIR/concepts/full_cov_shift/causal_report.json"; then
  echo "[$(timestamp)] skipping cov_shift causal validation"
else
  run_fg "$PYTHON_BIN" tools/causal_validate.py \
    --concept "$BASE_DIR/concepts/full_cov_shift" \
    --embeddings "$HUMAN_ACT_OUT" \
    --pb "$PB_FILE" \
    --max-samples "$CAUSAL_MAX_SAMPLES"
fi

if should_skip_file "$BASE_DIR/concepts/full_cluster_diff/report.json"; then
  echo "[$(timestamp)] skipping cluster_diff concepts"
else
  run_fg "$PYTHON_BIN" tools/discover_concepts.py \
    --embeddings-a "$LC0_ACT_OUT" \
    --embeddings-b "$HUMAN_ACT_OUT" \
    --out "$BASE_DIR/concepts/full_cluster_diff" \
    --method cluster_diff --k "$CONCEPT_K" \
    --max-samples "$CONCEPT_MAX_SAMPLES" \
    --patch --pb "$PB_FILE"
fi

if should_skip_file "$BASE_DIR/concepts/full_cluster_diff/causal_report.json"; then
  echo "[$(timestamp)] skipping cluster_diff causal validation"
else
  run_fg "$PYTHON_BIN" tools/causal_validate.py \
    --concept "$BASE_DIR/concepts/full_cluster_diff" \
    --embeddings "$HUMAN_ACT_OUT" \
    --pb "$PB_FILE" \
    --max-samples "$CAUSAL_MAX_SAMPLES"
fi

if should_skip_file "$BASE_DIR/concept_report.md"; then
  echo "[$(timestamp)] skipping concept report build"
else
  run_fg "$PYTHON_BIN" tools/build_concept_report.py --runs "$BASE_DIR/concepts" --out "$BASE_DIR/concept_report.md"
fi
if should_skip_file "$BASE_DIR/concepts_viz.html"; then
  echo "[$(timestamp)] skipping concept viz build"
else
  run_fg "$PYTHON_BIN" tools/concept_viz.py --runs "$BASE_DIR/concepts" --out "$BASE_DIR/concepts_viz.html"
fi

echo "[$(timestamp)] pipeline complete"
