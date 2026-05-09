# lc0jax

LC0 inference in JAX/Flax plus a Schut-style concept discovery pipeline.

This repository targets the LC0 BT4 network
`BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz` and provides:

- A JAX/Flax forward pass that mirrors LC0 outputs (policy + WDL + MLH).
- An ONNX oracle path for regression checks.
- A concept discovery workflow on LC0 activations.
- Hooks to continue training and export weights back into LC0 UCI.

## Layout

- `lc0jax/modeling/`: model, weights mapping, encoder, inference.
- `lc0jax/training/`: training utilities and LC0 training chunk parsing.
- `lc0jax/uci/`: interoperability with LC0 (oracle + export hooks).
- `lc0jax/interpretability/`: datasets, activation dumping, concept discovery.
- `tools/`: CLI utilities (export ONNX, run oracle, dump activations, download data).

## Reproducibility / Environment

Use `uv` to create isolated environments so experiments can be reproduced across machines.

- Create venv + install deps:
  - `uv venv .venv`
  - `uv pip install -e ".[dev]"`
- Reuse a shared wheel cache when moving between local and GCP machines:
  - `export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}`
- GPU (CUDA 13 wheels):
  - `uv pip install --python .venv/bin/python "jax[cuda13]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- CPU-only fallback:
  - `uv pip install --python .venv/bin/python "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html`

Record the LC0 version used for ONNX export and the BT4 model checksum (see `AGENTS.md`).

## Quick checks

- Oracle parity:
  - `python tools/compare_oracle_flax.py --onnx models/BT4.onnx --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --fens data/fens.txt`
- Activation dump:
  - `python tools/dump_activations.py --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --fens data/human.fens --out data/activations/human`
  - History-aware PGN records:
    `python tools/pgn_to_activation_records.py --pgn data/lichess/broadcasts.pgn --out data/lichess/broadcasts.records.jsonl`
  - Dump from history-aware records:
    `python tools/dump_activations.py --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --records data/lichess/broadcasts.records.jsonl --out data/activations/human_history`
  - For Schut-style square-aware runs, preserve spatial tokens:
    `python tools/dump_activations.py --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --fens data/human.fens --out data/activations/human_flat --activation-mode flat --store-token-activations`
- LC0 benchmark:
  - `python tools/lc0_benchmark.py --lc0 /tmp/lc0-src/build/release/lc0 --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --mode benchmark`
- Export back to LC0:
  - `python tools/export_pb.py --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --out models/BT4_exported.pb.gz`
  - ` /tmp/lc0-src/build/release/lc0 describenet --weights=models/BT4_exported.pb.gz`
- Concept discovery (additional methods):
  - `python tools/discover_concepts.py --embeddings-a data/activations/lc0 --embeddings-b data/activations/human_2000_rapid_classical --out data/concepts/cov_shift_2000 --method cov_shift --k 8 --max-samples 1000`
  - `python tools/discover_concepts.py --embeddings-a data/activations/lc0 --embeddings-b data/activations/human_2000_rapid_classical --out data/concepts/cluster_diff_2000 --method cluster_diff --k 8 --max-samples 1000`
- Dynamic sparse concepts from precomputed rollout pairs:
  - One-command GPU-oriented wrapper for high-strength PGN/FEN roots:
    `python tools/run_dynamic_gpu_pipeline.py --pgn data/pgn/tcec.pgn --pgn data/pgn/top_human.pgn --run-id dynamic_high_strength_4k --lc0 /tmp/lc0-src/build/release/lc0 --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --lc0-backend cuda --max-roots 30000 --max-pairs 4000 --nodes 800 --multipv 4 --activation-batch-size 64 --stop-after sweep`
    With `--pgn`, the wrapper writes filtered root-record JSONL with pre-root `history_fens` and passes `--root-records` into LC0 search. The MCTS builder reconstructs a short UCI move stack from consecutive history FENs when possible and records `root_history_reconstructed`. Plain `--fens` inputs remain supported for legacy/debug runs, but they only carry `[root_fen]` history.
    The wrapper writes `data/runs/<RUN_ID>/RUN_METADATA.md`, `dynamic_roots_manifest.json`, command logs, candidate roots or root records, LC0 rollout pairs, trajectory activations, materialized flat pairs, train/test splits, and the screened dynamic sweep. The manifest marks dry runs as planned, records output existence, and distinguishes fully history-faithful PGN roots from mixed PGN/FEN roots.
  - Runtime check for a new machine:
    `python tools/run_dynamic_gpu_pipeline.py --runtime-check-only`
  - Use `--dry-run` to record the planned commands without launching LC0 or JAX activation dumps. Use `--resume` to skip stages with wrapper completion markers, and use `--shard-count/--shard-index` to split LC0 MCTS extraction across workers. Sharded workers write isolated outputs under `data/runs/<RUN_ID>/shards/shard_XXX_of_YYY/`; run them with `--stop-after mcts` when distributing only the search step.
  - `--lc0-backend auto` resolves to `cuda` when JAX sees a GPU. Set `--lc0-backend cudnn` if LC0 was built with cuDNN, or `--lc0-backend none`/`eigen` for CPU fallback.
  - Prefer trajectory records for LC0 112-plane inputs so PV continuations keep rolling history:
    `python tools/dump_activations.py --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --records data/runs/<RUN_ID>/mcts_pairs/trajectory.records.jsonl --out data/runs/<RUN_ID>/activations/trajectory_flat --activation-mode flat --store-token-activations`
  - `python tools/materialize_mcts_pairs.py --pairs-jsonl data/runs/<RUN_ID>/mcts_pairs/pairs.jsonl --activations data/runs/<RUN_ID>/activations/trajectory_flat --out data/runs/<RUN_ID>/mcts_pairs/pairs.npz --mode flat`
  - `python tools/split_dynamic_pairs.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.npz --out-train data/runs/<RUN_ID>/mcts_pairs/pairs.train.npz --out-test data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --test-fraction 0.2 --seed 0`
    The split groups by root FEN without the fullmove counter; use repeated `--row-aligned-key` flags for any custom per-pair arrays.
  - `python tools/solve_dynamic_concepts.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.train.npz --out data/runs/<RUN_ID>/concepts/dynamic_sparse --mode flat`
    For large flat runs, use deterministic feature screening as a support-constrained approximation: it solves the same CVXPY objective on selected columns, then expands the solved direction back to the original feature dimension:
    `python tools/solve_dynamic_concepts.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.train.npz --out data/runs/<RUN_ID>/concepts/dynamic_sparse_screened --mode flat --max-features 2048`
  - Discover a library of dynamic concept families by clustering rollout differences and solving one sparse vector per cluster:
    `python tools/solve_dynamic_concept_families.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.train.npz --out data/runs/<RUN_ID>/concepts/dynamic_families --clusters 8 --max-features 2048 --bootstrap-count 8`
    This writes an aggregate `families.npz` plus per-family concept directories such as `family_000/` that can be passed to the held-out evaluation and prototype-selection tools.
  - Sweep screened solver settings and optional policy-margin alpha values:
    `python tools/sweep_dynamic_screening.py --train-pairs data/runs/<RUN_ID>/mcts_pairs/pairs.train.npz --test-pairs data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --out data/runs/<RUN_ID>/concepts/screening_sweep --max-features 1024,2048,4096,8192 --screening-methods abs_mean,mean_abs --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --policy-alphas 0.1,0.3,1.0,3.0`
  - `python tools/evaluate_dynamic_concept.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --out data/runs/<RUN_ID>/concepts/dynamic_sparse/heldout_eval_report.json --split-name test`
    This evaluates `raw_direction` by default so held-out margin metrics are comparable to the solver report.
  - `python tools/select_dynamic_prototypes.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.train.npz --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --out data/runs/<RUN_ID>/concepts/dynamic_sparse/prototypes_report.json --top-k 32 --random-count 32 --split-name train`
    The selector auto-detects reversed concepts from `<concept>/report.json` and preserves row-aligned metadata in selected rows.
  - `python tools/export_teachability_curriculum.py --prototypes data/runs/<RUN_ID>/concepts/dynamic_sparse/prototypes_report.json --out data/runs/<RUN_ID>/concepts/dynamic_sparse/teachability_curriculum.jsonl`
  - `python tools/dynamic_concept_baselines.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --out data/runs/<RUN_ID>/concepts/dynamic_sparse/baselines_report.json`
  - `python tools/dynamic_policy_margin.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --out data/runs/<RUN_ID>/concepts/dynamic_sparse/policy_margin_report.json --control-count 16 --control-kind random`
    Use `--control-count 0` for the learned-direction-only report; random or shuffled controls calibrate whether the learned patch effect is larger than same-norm control directions.
  - `python tools/build_dynamic_concept_report.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --families data/runs/<RUN_ID>/concepts/dynamic_families/report.json --out data/runs/<RUN_ID>/concepts/dynamic_sparse/report.md`
    The optional `--families` report adds a concept-family summary table. It only fills per-family held-out, prototype, and policy-margin columns after you run those validation tools for each `family_XXX/` directory you want to compare.
- Novelty filtering:
  - `python tools/filter_novel_concepts.py --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --machine-embeddings data/runs/<RUN_ID>/activations/lc0_flat --human-embeddings data/runs/<RUN_ID>/activations/human_flat --out data/runs/<RUN_ID>/concepts/dynamic_sparse/novelty_report.json`
- Causal validation (patching across many positions):
  - `python tools/causal_validate.py --concept data/concepts/mean_diff_2000 --embeddings data/activations/human_2000_rapid_classical --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --max-samples 256`
- Concept report:
  - `python tools/build_concept_report.py --runs data/concepts --out data/concept_report.md`
- Concept visualizations (HTML):
  - `python tools/concept_viz.py --runs data/concepts --out data/concepts_viz.html`
- UCI text UI (play vs LC0):
  - `python tools/uci_ui.py --lc0 /tmp/lc0-src/build/release/lc0 --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --analysis --movetime-ms 200`
- Elo bench (A vs B weights):
  - `python tools/elo_bench.py --lc0 /tmp/lc0-src/build/release/lc0 --weights-a models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --weights-b models/BT4_exported.pb.gz --games 20 --nodes 800`

## One-command data pipeline

Run the full data pipeline (broadcasts download + LC0 chunks + activation dumps + concept runs).
Outputs are stored under `data/runs/<RUN_ID>/` by default:

```
./tools/run_full_pipeline.sh
```

Skip existing outputs (resume mode):
```
SKIP_EXISTING=1 ./tools/run_full_pipeline.sh
```

Environment overrides (examples):
- `RUN_ID=2026-02-02_full SKIP_EXISTING=1 ./tools/run_full_pipeline.sh`
- `BROADCAST_GAMES=20000 BROADCAST_NB=2000 BROADCAST_MAX_BROADCASTS=2000 ./tools/run_full_pipeline.sh`
- `LC0_MAX_POSITIONS=200000 CONCEPT_MAX_SAMPLES=200000 ./tools/run_full_pipeline.sh`
- `BROADCAST_SLEEP=1.0 BROADCAST_MAX_RETRIES=8 BROADCAST_RETRY_BACKOFF=10 ./tools/run_full_pipeline.sh`
- `FILTER_HUMAN=1 FILTER_MIN_PLY=12 FILTER_MAX_PHASE=0.8 ./tools/run_full_pipeline.sh`
- `DISAGREE_FILTER=1 DISAGREE_LC0_BIN=/tmp/lc0-src/build/release/lc0 ./tools/run_full_pipeline.sh`
- `HISTORY_HUMAN_RECORDS=0 ./tools/run_full_pipeline.sh` to force FEN-only human activation dumps.

Large LC0 search, full activation dumps, SVD novelty sweeps on large matrices, and teachability
training should run on GCP. Keep local runs to unit tests, small smoke datasets, and shape checks.

For non-GCP GPU machines without GCP authentication, use
`NON_GCP_GPU_RUNBOOK.md`. It lists the public HTTPS input URLs, local output
locations, CPU staging workflow, GPU setup, LC0 build notes, smoke test, sharded
MCTS run, merge path, and artifact packaging commands.

## Filtering positions

Use `tools/filter_fens.py` to remove opening positions or tablebase-like endgames before activation dumps.
Phase is normalized to `[0, 1]` with `1.0` = opening and `0.0` = pure endgame.

- Example:
  - `python tools/filter_fens.py --fens data/lichess/broadcasts.fens --out data/lichess/broadcasts.filtered.fens --min-ply 12 --max-phase 0.8 --min-pieces 8 --dedupe`
- LC0 search vs raw policy disagreement (novelty filter):
  - `python tools/filter_fens_disagreement.py --fens data/lichess/broadcasts.filtered.fens --out data/lichess/broadcasts.disagree.fens --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --lc0 /tmp/lc0-src/build/release/lc0 --nodes 800`
  - Use `--onnx models/BT4.onnx` to avoid JAX compilation when generating the raw policy logits.
  - For parallel runs: `--shard-count 4 --shard-index 0` (run 0..3 in separate shells and merge outputs).
  - To resume manually: re-run with `--start-line N` and `--append`, optionally `--state-file` to track progress.
- LC0 MCTS rollout-pair metadata for dynamic concepts:
  - Preferred history-faithful input:
    `python tools/build_mcts_pairs.py --root-records data/lichess/broadcasts.records.jsonl --out-jsonl data/runs/<RUN_ID>/mcts_pairs/pairs.jsonl --out-trajectory-records data/runs/<RUN_ID>/mcts_pairs/trajectory.records.jsonl --out-trajectory-fens data/runs/<RUN_ID>/mcts_pairs/trajectory.fens --lc0 /tmp/lc0-src/build/release/lc0 --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --nodes 800 --multipv 4 --max-pairs 100`
  - Legacy/debug FEN input:
    `python tools/build_mcts_pairs.py --fens data/lichess/broadcasts.filtered.fens --out-jsonl data/runs/<RUN_ID>/mcts_pairs/pairs.jsonl --out-trajectory-records data/runs/<RUN_ID>/mcts_pairs/trajectory.records.jsonl --out-trajectory-fens data/runs/<RUN_ID>/mcts_pairs/trajectory.fens --lc0 /tmp/lc0-src/build/release/lc0 --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --nodes 800 --multipv 4 --max-pairs 100`

## Data sources

- Human: use Lichess monthly standard rated PGNs, then filter to high-rated classical games.
  - Download latest month: `python tools/download_data.py lichess --out-dir data/lichess --decompress`
  - Or stream a filtered sample without the full download:
    `python tools/download_data.py lichess-sample --out-pgn data/lichess/lichess_sample.pgn --max-games 2000 --min-elo 2000 --time-class rapid --time-class classical --rated`
  - Example filtered outputs: `data/lichess/lichess_sample_2000_rapid_classical.pgn` + `.fens`
  - Optional 5k-position subset for faster activation dumps: `data/lichess/lichess_sample_2000_rapid_classical_5k.fens`
  - Note: 2400+ classical is sparse in the monthly dump; you may need to sample far more games (or use a top-player API pull) to reach 1k games.
  - `python tools/filter_pgn.py --pgn data/lichess/lichess_db_standard_rated_YYYY-MM.pgn --out-fens data/human_rapid_classical_top.fens --max-games 1000 --min-elo 2000 --time-class rapid --time-class classical --rated`
- Human (OTB-style via Lichess broadcasts API):
  - `python tools/download_data.py lichess-broadcasts --out-pgn data/lichess/broadcasts.pgn --out-fens data/lichess/broadcasts.fens --max-games 1000 --min-elo 2400`
- Human (tactics via Lichess puzzles DB):
  - `python tools/download_data.py lichess-puzzles --out-fens data/lichess/puzzles_2500.fens --min-rating 2500`
  - By default, the puzzle loader applies the first move so the FEN represents the position shown to the solver (use `--raw-fen` to keep the original).
- Computer championship or top-human PGNs:
  - Store source PGNs under `data/pgn/` and pass one or more files directly to `tools/run_dynamic_gpu_pipeline.py --pgn ...`. The wrapper samples positions from every mainline, preserves pre-root LC0 history in root records, filters openings/endgames, and can shard candidate roots before LC0 search.
- Computer: use LC0 training chunks or generate selfplay. Training chunks can be converted to one-ply PGNs/FENs:
  - Download latest LC0 chunk tar: `python tools/download_data.py lc0-chunks --out-dir data/lc0-training --count 1 --min-size 1000000`
  - `python tools/chunks_to_pgn.py --chunk data/chunk1.gz --chunk data/chunk2.gz --out-pgn data/lc0_chunk.pgn --out-fens data/lc0_chunk.fens --max-positions 1000`
- Reference manifests:
  - Human reference contract:
    `python tools/write_reference_manifest.py --kind human --name human_reference_v1 --source-type lichess_standard --source-url https://database.lichess.org/standard/ --input data/pgn/lichess_2400_rapid_classical.pgn --output data/pgn/lichess_2400_rapid_classical.fens --out data/manifests/human_reference_v1.json --min-elo 2400 --time-class rapid --time-class classical --rated --dedupe-key "board_fen side castling ep" --split-key game_id --exclude tablebase_7_or_less --exclude variant_nonstandard`
  - Machine reference contract:
    `python tools/write_reference_manifest.py --kind machine --name machine_reference_v1 --source-type tcec --source-url https://github.com/TCEC-Chess/tcecgames/releases --input data/pgn/tcec_compact_all.pgn --out data/manifests/machine_reference_v1.json --min-ply 18 --min-phase 0.25 --max-phase 0.85 --dedupe-key "board_fen side castling ep" --split-key game_id --exclude tablebase_7_or_less`
  - Use `--no-checksum` for very large staged files when hashing would be too slow; otherwise the manifest records local file SHA256s. The writer fails if a listed local file is missing unless you pass `--allow-missing` for a planned manifest.

Note: LC0 training chunks are not sequential game records; the PGN conversion emits one-ply games per position.
See `TESTS.md` for the rationale behind the current test suite.
Note: Lichess PGNs are distributed as `.zst`; `python tools/download_data.py lichess --decompress` requires the `zstandard` Python package.

## Roadmap

See `PLANS.md` for the execution plan, including the export-to-LC0 milestone (JAX weights → `.pb.gz` for UCI).
