# Non-GCP GPU Runbook

This runbook is for running the LC0 dynamic concept pipeline on a GPU machine
without authenticating to GCP. All required inputs are public HTTPS downloads or
local files copied from a staging box. There is no required write bucket.

## Public Input URLs

Use these as source URLs. They do not require GCP credentials.

| Purpose | URL |
| --- | --- |
| Target BT4 weights | `https://storage.lczero.org/files/networks-contrib/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz` |
| LC0 source / releases | `https://github.com/LeelaChessZero/lc0/releases` |
| LC0 v0.32.1 source tarball | `https://github.com/LeelaChessZero/lc0/archive/refs/tags/v0.32.1.tar.gz` |
| TCEC release page | `https://github.com/TCEC-Chess/tcecgames/releases` |
| TCEC latest compact archive | `https://github.com/TCEC-Chess/tcecgames/releases/latest/download/TCEC-everything-compact.zip` |
| TCEC current pinned compact archive | `https://github.com/TCEC-Chess/tcecgames/releases/download/S28-final/TCEC-everything-compact.zip` |
| Lichess standard index | `https://database.lichess.org/standard/` |
| Lichess standard list | `https://database.lichess.org/standard/list.txt` |
| Lichess standard checksums | `https://database.lichess.org/standard/sha256sums.txt` |
| Current latest Lichess standard dump, checked 2026-04-28 | `https://database.lichess.org/standard/lichess_db_standard_rated_2026-03.pgn.zst` |
| Lichess broadcast database | `https://database.lichess.org/#broadcasts` |
| Current latest Lichess broadcast dump, checked 2026-04-28 | `https://database.lichess.org/broadcast/lichess_db_broadcast_2026-03.pgn.zst` |
| Lichess puzzle dump | `https://database.lichess.org/lichess_db_puzzle.csv.zst` |
| LC0 self-play training data index | `https://storage.lczero.org/files/training_data/` |
| JAX CUDA wheel index | `https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` |

The preferred first dataset is TCEC compact PGNs plus optionally Lichess
broadcast or high-rated rapid/classical samples. The repo tools can also stream
from the latest Lichess standard dump without storing the full 30 GB archive.

## Output Locations

Write outputs locally on the GPU box:

- Main run directory: `data/runs/<RUN_ID>/`
- Sharded run work directories: `data/runs/<RUN_ID>/shards/shard_XXX_of_YYY/`
- Command log per work directory: `commands.jsonl`
- Reproducibility metadata per work directory: `RUN_METADATA.md`
- Locked dynamic-root manifest per work directory: `dynamic_roots_manifest.json`
- Summary per work directory: `run_summary.json`
- Recommended artifact bundle: `artifacts/<RUN_ID>_artifacts.tar.zst`

`dynamic_roots_manifest.json` records whether the run is planned, partial, or
completed, which outputs existed when the manifest was written, and whether the
root set is fully history-faithful or contains root-only FEN records.

There is no project-owned public write bucket. If you want to upload results,
use your own destination, for example:

- `s3://<your-bucket>/schutpaper/runs/<RUN_ID>/`
- `gs://<your-bucket>/schutpaper/runs/<RUN_ID>/`
- `rclone:<remote>/schutpaper/runs/<RUN_ID>/`

Do not require GCP authentication unless your chosen destination is a private
GCS bucket. A simple `scp` or `rsync` copy of `artifacts/<RUN_ID>_artifacts.tar.zst`
back to a workstation is enough.

## CPU Staging Box

Use the CPU box for disk-heavy preparation and dependency caching. Do not assume
a copied `.venv` will work on the GPU box; copy the `uv` cache and recreate the
venv on the target machine.

```bash
git clone https://github.com/LiquidGunay/lc0_bridging_the_gap.git schutpaper
cd schutpaper

export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python tools/download_model.py
sha256sum models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz
```

Download TCEC compact PGNs:

```bash
mkdir -p data/pgn
curl -L \
  -o data/pgn/TCEC-everything-compact.zip \
  https://github.com/TCEC-Chess/tcecgames/releases/download/S28-final/TCEC-everything-compact.zip
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

out = Path("data/pgn/tcec")
out.mkdir(parents=True, exist_ok=True)
with ZipFile("data/pgn/TCEC-everything-compact.zip") as zf:
    zf.extractall(out)
print("Extracted", len(list(out.rglob("*.pgn"))), "PGN files")
PY
find data/pgn/tcec -name '*.pgn' -print0 | sort -z | \
  xargs -0 cat > data/pgn/tcec_compact_all.pgn
```

Optionally stream a high-rated Lichess rapid/classical sample:

```bash
.venv/bin/python tools/download_data.py lichess-sample \
  --out-pgn data/pgn/lichess_2400_rapid_classical.pgn \
  --out-fens data/pgn/lichess_2400_rapid_classical.fens \
  --max-games 1000 \
  --min-elo 2400 \
  --time-class rapid \
  --time-class classical \
  --rated \
  --progress-every 100
```

Prepare a dry-run plan. With `--pgn`, this writes filtered candidate root
records with pre-root `history_fens`, plus command metadata, but does not
require LC0 or a GPU:

```bash
RUN_ID=dynamic_high_strength_4k

.venv/bin/python tools/run_dynamic_gpu_pipeline.py \
  --run-id "$RUN_ID" \
  --pgn data/pgn/tcec_compact_all.pgn \
  --pgn data/pgn/lichess_2400_rapid_classical.pgn \
  --dry-run \
  --stop-after mcts \
  --max-roots 30000 \
  --max-pairs 4000 \
  --nodes 800 \
  --multipv 4 \
  --shard-count 8 \
  --shard-index 0
```

Repeat the dry-run with `--shard-index 1` through `7` if you want all shard root
record files prepared before copying to the GPU box. Plain FEN lists can still
be passed with `--fens` for debugging, but those roots do not carry PGN history.

Bundle the staged repo and uv cache:

```bash
mkdir -p artifacts
tar --exclude .git --exclude .venv \
  -I 'zstd -T0 -3' \
  -cf artifacts/schutpaper_gpu_stage.tar.zst \
  .
tar -I 'zstd -T0 -3' \
  -cf artifacts/uv_cache.tar.zst \
  -C "$HOME/.cache" uv
```

Copy both archives to the GPU box with `scp`, `rsync`, or your own object-store
destination.

## GPU Box Setup

Unpack the staged repo:

```bash
mkdir -p ~/schutpaper
tar -I zstd -xf schutpaper_gpu_stage.tar.zst -C ~/schutpaper
cd ~/schutpaper

mkdir -p ~/.cache
tar -I zstd -xf uv_cache.tar.zst -C ~/.cache

export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
uv pip install --python .venv/bin/python "jax[cuda13]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify the GPU runtime:

```bash
nvidia-smi || true
.venv/bin/python tools/run_dynamic_gpu_pipeline.py --runtime-check-only
```

The runtime check must show `jax_gpu_visible: true` before full activation dumps.

## LC0 On The GPU Box

The official LC0 v0.32.1 release has no Linux binary asset, so build LC0 on the
GPU box or on a machine with matching CUDA/cuDNN development libraries. Keep the
artifact at `/tmp/lc0-src/build/release/lc0` unless you pass a different `--lc0`.

```bash
cd /tmp
git clone --branch v0.32.1 --depth 1 --recurse-submodules \
  https://github.com/LeelaChessZero/lc0 lc0-src
cd /tmp/lc0-src
./build.sh release
/tmp/lc0-src/build/release/lc0 --help | head -n 5
```

If CUDA/cuDNN build support is unavailable, build the CPU backend first and run
a small correctness smoke with `--lc0-backend eigen`. Use `cuda` or `cudnn` only
after LC0 accepts that backend in a short benchmark:

```bash
cd ~/schutpaper
.venv/bin/python tools/lc0_benchmark.py \
  --lc0 /tmp/lc0-src/build/release/lc0 \
  --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz \
  --mode benchmark
```

## Small GPU Smoke

Run one small end-to-end shard before the expensive job:

```bash
RUN_ID=dynamic_gpu_smoke

.venv/bin/python tools/run_dynamic_gpu_pipeline.py \
  --run-id "$RUN_ID" \
  --pgn data/pgn/tcec_compact_all.pgn \
  --lc0 /tmp/lc0-src/build/release/lc0 \
  --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz \
  --lc0-backend cuda \
  --max-roots 100 \
  --max-pairs 10 \
  --nodes 200 \
  --multipv 4 \
  --activation-batch-size 16 \
  --max-features 512 \
  --screening-methods abs_mean \
  --stop-after sweep
```

If LC0 was built with cuDNN rather than CUDA, replace `--lc0-backend cuda` with
`--lc0-backend cudnn`. If the GPU backend fails, retry with `--lc0-backend eigen`
to separate LC0 backend issues from pipeline issues.

## Full Sharded Run

Run one shard per shell, tmux pane, or worker. Sharded workdirs are isolated.
Use `--stop-after mcts` for the distributed search pass.

```bash
RUN_ID=dynamic_high_strength_4k
SHARDS=8
IDX=0

.venv/bin/python tools/run_dynamic_gpu_pipeline.py \
  --run-id "$RUN_ID" \
  --pgn data/pgn/tcec_compact_all.pgn \
  --pgn data/pgn/lichess_2400_rapid_classical.pgn \
  --lc0 /tmp/lc0-src/build/release/lc0 \
  --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz \
  --lc0-backend cuda \
  --max-roots 30000 \
  --max-pairs 500 \
  --nodes 800 \
  --multipv 4 \
  --activation-batch-size 64 \
  --max-features 2048 \
  --screening-methods abs_mean \
  --shard-count "$SHARDS" \
  --shard-index "$IDX" \
  --stop-after mcts \
  --resume
```

After MCTS shards finish, choose one of two paths:

1. Continue each shard independently through `--stop-after sweep`. This is the
   simplest path and produces one concept sweep per shard.
2. Merge shard `pairs.jsonl` and `trajectory.records.jsonl` into a combined
   workdir, then run activations/materialization/split/sweep once. Use this
   path when you want one concept over all collected pairs.

The merge path is manual for now:

```bash
RUN_ID=dynamic_high_strength_4k
MERGED=data/runs/${RUN_ID}/merged
mkdir -p "$MERGED/mcts_pairs"
cat data/runs/${RUN_ID}/shards/shard_*_of_*/mcts_pairs/pairs.jsonl \
  > "$MERGED/mcts_pairs/pairs.jsonl"
cat data/runs/${RUN_ID}/shards/shard_*_of_*/mcts_pairs/trajectory.records.jsonl \
  > "$MERGED/mcts_pairs/trajectory.records.jsonl"

.venv/bin/python tools/dump_activations.py \
  --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz \
  --records "$MERGED/mcts_pairs/trajectory.records.jsonl" \
  --out "$MERGED/activations/trajectory_flat" \
  --layer trunk \
  --activation-mode flat \
  --store-token-activations \
  --batch-size 64 \
  --shard-size 4096 \
  --count-fens \
  --progress-every 1000

.venv/bin/python tools/materialize_mcts_pairs.py \
  --pairs-jsonl "$MERGED/mcts_pairs/pairs.jsonl" \
  --activations "$MERGED/activations/trajectory_flat" \
  --out "$MERGED/mcts_pairs/pairs.npz" \
  --mode flat

.venv/bin/python tools/split_dynamic_pairs.py \
  --pairs "$MERGED/mcts_pairs/pairs.npz" \
  --out-train "$MERGED/mcts_pairs/pairs.train.npz" \
  --out-test "$MERGED/mcts_pairs/pairs.test.npz" \
  --test-fraction 0.2 \
  --seed 0

.venv/bin/python tools/sweep_dynamic_screening.py \
  --train-pairs "$MERGED/mcts_pairs/pairs.train.npz" \
  --test-pairs "$MERGED/mcts_pairs/pairs.test.npz" \
  --out "$MERGED/concepts/screening_sweep" \
  --mode flat \
  --max-features 2048 \
  --screening-methods abs_mean \
  --skip-policy-margin

.venv/bin/python tools/solve_dynamic_concept_families.py \
  --pairs "$MERGED/mcts_pairs/pairs.train.npz" \
  --out "$MERGED/concepts/dynamic_families" \
  --clusters 8 \
  --max-features 2048 \
  --bootstrap-count 8

.venv/bin/python tools/dynamic_policy_margin.py \
  --pairs "$MERGED/mcts_pairs/pairs.test.npz" \
  --concept "$MERGED/concepts/dynamic_families/family_000" \
  --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz \
  --out "$MERGED/concepts/dynamic_families/family_000/policy_margin_report.json" \
  --max-pairs 64 \
  --control-count 16 \
  --control-kind random
```

## Package Results

Package the run directory and copy it off the GPU box:

```bash
RUN_ID=dynamic_high_strength_4k
mkdir -p artifacts
tar -I 'zstd -T0 -3' \
  -cf "artifacts/${RUN_ID}_artifacts.tar.zst" \
  "data/runs/${RUN_ID}"
```

If using a private bucket, upload the artifact with your own credentials and
provider CLI. Otherwise use `scp` or `rsync`.

## Expected First Review Items

After the GPU run finishes, inspect:

- `data/runs/<RUN_ID>/**/RUN_METADATA.md`
- `data/runs/<RUN_ID>/**/commands.jsonl`
- Number of kept MCTS records in `mcts_pairs/pairs.jsonl`
- `mcts_pairs/pairs.npz` shape and train/test split sizes
- `concepts/screening_sweep/summary.md`
- `concepts/dynamic_families/report.json`
- `concepts/dynamic_families/family_000/policy_margin_report.json`
- Whether `abs_mean_2048` still beats smaller/larger feature caps
- Whether policy-margin should be rerun on the strongest held-out concept
