# LC0JAX + Schut-Style Concepts for LC0


This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must be kept up to date as work proceeds.

This file is `PLANS.md` in the repository root and must be maintained in accordance with `ExecPLANS.md`.

## Purpose / Big Picture


The goal is to build an open-source, reproducible pipeline that can load a real LC0 BT4 network, run LC0-style inference in JAX/Flax on chess positions, and then reproduce the Schut-style concept discovery workflow on LC0 representations. After this plan is implemented, a user can: (1) load the specific BT4 model file, encode a FEN into LC0 input planes, run a forward pass that produces policy and value/WDL outputs, and verify outputs against an ONNX oracle; and (2) run a concept discovery workflow that compares human versus LC0 self-play positions, extracts activation embeddings, produces prototype positions, and validates concept directions via activation patching with measurable changes in move logits or value. The proof of success is that the provided CLI tools run end-to-end on a small test FEN set and on a larger dataset shard, with reproducible, human-inspectable outputs.

## Progress


- [x] (2026-01-29 00:00Z) Created initial ExecPlan in `PLANS.md` with target BT4 network, simplified repo structure, and Schut-style concept workflow.
- [x] (2026-01-29 18:15Z) Bootstrap repo structure, dependencies, and model artifact download script (completed: repo skeleton, tool stubs, `tools/download_model.py`, created `.venv` via `uv`, installed core deps, built LC0 v0.32.1 from source, exported ONNX).
- [x] (2026-01-29 18:15Z) Implement ONNX oracle conversion + runner with a deterministic FEN test set (completed: built LC0 v0.32.1, exported ONNX via `lc0 leela2onnx`, ran oracle on `data/fens.txt` with encoder, confirmed input/output shapes).
- [x] (2026-01-29 18:45Z) Implement LC0 input planes, policy mapping, and legality masking with tests (completed: LC0 policy move list extraction, attention policy mapping table, legality mask, encoder port with canonical/classical formats, encoder/policy tests, ONNX oracle shape/NaN test; parity against ONNX on `data/fens.txt` confirms encoder alignment).
- [x] (2026-01-29 18:30Z) Parse .pb.gz weights, implement BT4 Flax model, and reach loose parity against the oracle (completed: protobuf bindings generated, weight tensor decoding and mapping into a BT4 param dict, Flax/JAX forward implemented, `tools/compare_oracle_flax.py` reports perfect top-1/top-5/value correlation on `data/fens.txt`).
- [x] (2026-01-29 17:45Z) Prepare protobuf schema for weights parsing (completed: vendored `lc0jax/proto/net.proto` and `lc0jax/proto/chunk.proto` from lczero-common, added `tools/gen_proto.py`, generated `net_pb2.py`/`chunk_pb2.py` via `grpcio-tools`, implemented full tensor decoding and metadata capture).
- [x] (2026-02-01 12:20Z) Add activation capture, dataset pipeline, concept discovery, prototypes, and causal validation (completed: PGN-to-FEN conversion utilities, PGN filtering by rating/time control, activation dump tool and shard format, mean-difference concept discovery with prototype extraction, additional concept methods, causal validation across multiple directions, and a generated `data/concept_report.md`).
- [x] (2026-02-01 06:05Z) Installed CUDA-enabled JAX (`jax[cuda13]`) in `.venv` and verified the GPU backend is active.
- [x] (2026-02-01 06:20Z) Restructured code into `lc0jax/modeling`, `lc0jax/interpretability`, `lc0jax/training`, and `lc0jax/uci` with compatibility wrappers and updated tools/imports.
- [x] (2026-02-01 06:30Z) Added training chunk parser plus a chunk→PGN/FEN converter (one-ply PGNs per position).
- [x] (2026-02-01 07:20Z) Added a download helper for Lichess standard PGNs and LC0 training tarballs, plus tests for dataset parsing and chunk conversion; added `zstandard` dependency for `.zst` decompression.
- [x] (2026-02-01 07:35Z) Ran `pytest -q` after adding dataset/chunk tests; all 14 tests passed.
- [x] (2026-02-01 07:45Z) Added `tools/lc0_benchmark.py` to wrap LC0 benchmark modes (benchmark/bench/backendbench) for quick performance baselines.
- [x] (2026-02-01 08:05Z) Added a streaming Lichess sampler (`download_data.py lichess-sample`) plus stream-filter tests; this avoids full monthly downloads when only ~1k games are needed.
- [x] (2026-02-01 08:20Z) Added progress logging for streamed PGN sampling and re-ran `pytest -q` (all tests pass).
- [x] (2026-02-01 08:45Z) Streamed 200k rated Lichess games into `data/lichess/lichess_sample_raw_200k.pgn` and filtered to classical 2400+ (only 7 games matched; see Surprises).
- [x] (2026-02-01 08:50Z) Downloaded and extracted LC0 training tar `training-run2--20251215-1017.tar`, converted 50 chunks into 1,000 positions (`data/lc0-training/lc0_sample.*`).
- [x] (2026-02-01 08:55Z) Ran activation dumps for human and LC0 samples and re-ran `pytest -q` (all tests pass).
- [x] (2026-02-01 08:58Z) Ran LC0 benchmark with the BT4 weights (Eigen backend).
- [x] (2026-02-01 09:10Z) Added rapid+classical time-class support in PGN filtering and added `TESTS.md` describing test rationale.
- [x] (2026-02-01 09:15Z) Generated rapid+classical 2400+ human sample files and activation shards under `data/activations/human_rapid_classical`.
- [x] (2026-02-01 10:20Z) Lowered the human threshold to 2000+ rapid/classical to reach 1,000 games and generated `data/lichess/lichess_sample_2000_rapid_classical.*`.
- [x] (2026-02-01 10:42Z) Created a 5k-position subset and dumped activations to `data/activations/human_2000_rapid_classical`, then ran concept discovery against LC0 activations.
- [x] (2026-02-01 10:45Z) Re-ran `pytest -q` after dataset/filter changes; all tests pass.
- [x] (2026-02-01 10:46Z) Ran a second concept discovery pass (`whitened_mean_diff`) with patching output under `data/concepts/whitened_mean_diff_2000`.
- [x] (2026-02-01 10:55Z) Implemented BT4 `.pb.gz` export tooling and verified `lc0 describenet` on `models/BT4_exported.pb.gz`.
- [x] (2026-02-01 11:10Z) Parity check: `tools/compare_oracle_flax.py` against `models/BT4_exported.pb.gz` reports perfect agreement on `data/fens.txt`.
- [x] (2026-02-01 11:30Z) Added covariance-shift and clustering concept methods with multi-vector patching; ran both on LC0 vs human embeddings.
- [x] (2026-02-01 12:10Z) Added Elo bench tool, UCI text UI, and HTML concept visualizations; generated concept visualizations and documented usage.
- [x] (2026-02-01 12:30Z) Ran a quick Elo bench (2 games, nodes=200) comparing original vs exported weights and generated `data/concepts_viz.html`.
- [x] (2026-02-04 04:52Z) Switched the full-scale activation dump to the broadcast 2400+ classical dataset (36,039 FENs) and completed activation dumps under `data/runs/2026-02-02_full/activations`.
- [x] (2026-02-01 13:15Z) Added progress logging for activation dumps and a Lichess broadcast API downloader for OTB-style human data.
- [x] (2026-02-01 14:10Z) Added `tools/run_full_pipeline.sh` to run the full data pipeline in one command, with parallel downloads and sequential compute steps.
- [x] (2026-02-02 00:00Z) Switched pipeline outputs to `data/runs/<RUN_ID>/` with `SKIP_EXISTING=1` resume mode; moved existing data into `data/runs/legacy_2026-02-02`.
- [x] (2026-02-04 04:52Z) Fixed concept report/viz tooling to accept a parent run directory; regenerated `concept_report.md` and `concepts_viz.html` for `data/runs/2026-02-02_full`.
- [x] (2026-02-04 05:12Z) Added FEN filtering utilities (`tools/filter_fens.py`) plus optional pipeline hooks to drop opening positions before activation dumps.
- [x] (2026-02-04 05:40Z) Added a Lichess puzzles downloader (rating >= 2500 default) and an LC0 search-vs-policy disagreement filter tool.
- [x] (2026-04-27 00:00Z) Reviewed `REPO_AUDIT_AND_NEXT_STEPS.md` and recorded the current Schut-parity status in `IMPLEMENTATION_STATUS_AND_NEXT_WORK.md`.
- [x] (2026-04-27 00:00Z) Added square-aware activation projection modes, optional raw token/policy-logit storage, reusable paired-difference sparse solving, dynamic rollout-difference aggregation, and SVD novelty filtering.
- [x] (2026-04-27 00:00Z) Added a metadata-first LC0 MultiPV rollout-pair extractor (`tools/build_mcts_pairs.py`) that writes root, best PV, selected subpar PVs, scores, trajectory FENs, and history-aware trajectory activation records.
- [x] (2026-04-27 00:00Z) Verified the Schut-parity infrastructure changes with the full test suite (`33 passed`) and CLI import smoke tests for the new tools.
- [x] (2026-04-27 00:00Z) Added `tools/materialize_mcts_pairs.py` to join rollout-pair JSONL with trajectory activation shards and write solver-ready `pairs.npz` difference matrices, using stable activation keys when available.
- [x] (2026-04-27 00:00Z) Added history-aware PGN activation records and `dump_activations.py --records` so PGN-derived inputs can pass rolling history boards into LC0 encoding.
- [x] (2026-04-27 00:00Z) Updated `tools/run_full_pipeline.sh` to default to history-aware human activation records when the broadcast PGN is present.
- [x] (2026-04-27 14:10Z) Ran the first small GCP dynamic-concept smoke pipeline from LC0 MultiPV pairs through flat activation dump, `pairs.npz` materialization, sparse solve, and novelty report under `data/runs/gcp_dynamic_smoke_20260427`.
- [x] (2026-04-27 14:35Z) Re-ran the GCP dynamic-concept smoke with history-aware trajectory records under `data/runs/gcp_dynamic_smoke_records_20260427`; activation shards carried `activation_keys`, materialization produced a `(1, 65536)` difference matrix, the sparse solve was `optimal`, and the novelty smoke wrote an accepted toy vector.
- [x] (2026-04-27 14:55Z) Added dynamic concept markdown report cards from `pairs.npz`, solver reports, and novelty reports.
- [x] (2026-04-27 15:05Z) Added random sparse, shuffled-label, and optional shuffled-solve baselines for dynamic concept runs.
- [x] (2026-04-27 15:45Z) Added dynamic policy-margin patch reports and flat token-direction patch support for best-vs-subpar root move margins.
- [x] (2026-04-27 16:20Z) Added root-grouped train/test splitting for dynamic `pairs.npz` files before scaled held-out evaluation.
- [x] (2026-04-27 16:40Z) Added explicit held-out dynamic direction evaluation reports for train/test split workflows.
- [x] (2026-04-27 17:00Z) Added dynamic prototype and random-control selection reports as teachability curriculum inputs.
- [x] (2026-04-27 17:15Z) Added JSONL teachability curriculum export from dynamic prototype/control reports.
- [x] (2026-04-27 18:50Z) Ran larger GCP dynamic validation `data/runs/gcp_dynamic_large_20260427_174945` on `pipeline-vm`: 800 LC0 nodes, MultiPV 4, 40 kept rollout-pair records from 49 scanned roots, 948 history-aware trajectory records, 93 materialized dynamic differences, grouped 72 train / 21 held-out split, and completed a mean-pooled end-to-end report after identifying flat sparse solve runtime as the next bottleneck.
- [x] (2026-04-27 19:10Z) Added a screened flat dynamic solver path (`tools/solve_dynamic_concepts.py --max-features`) that preserves the exact sparse CVXPY objective on a deterministic feature subset and expands directions back to the original feature dimension; validated it on the preserved `(93, 65536)` GCP flat pairs with a 2048-feature screen.
- [ ] Add teachability evaluation with a weaker LC0 checkpoint or student network and random-prototype baselines.

## Surprises & Discoveries


- Observation: The system Python lacks `venv`/`pip` support (no `python3-venv` package), and `sudo` is unavailable. Creating a local venv or installing pip failed, which blocks generating protobuf bindings via `protoc` or `grpcio-tools` without another workaround.
  Evidence: `python3 -m venv .venv` reports that `ensurepip` is unavailable and suggests installing `python3.12-venv`; `sudo apt-get` is unavailable without a password.
- Observation: `uv` is available and can create a local venv and install packages, which unblocked `grpcio-tools` and protobuf generation.
  Evidence: `uv venv .venv` succeeded, `uv pip install ... grpcio-tools` succeeded, and `grpc_tools.protoc` generated `lc0jax/proto/net_pb2.py`.
- Observation: The LC0 binary does not support `--version`; the version is shown in the startup banner (`lc0 --help` prints it).
  Evidence: `/tmp/lc0-src/build/release/lc0 --version` reports an unknown flag, while `lc0 --help` prints `v0.32.1 built Jan 29 2026`.
- Observation: The BT4 model download requires a User-Agent header; plain urllib requests returned HTTP 403.
  Evidence: `tools/download_model.py` initially failed with HTTP 403 until a User-Agent header was added.
- Observation: The v0.32.1 LC0 release assets do not include a Linux binary.
  Evidence: GitHub release assets list only Android and Windows builds.
- Observation: Editable install pulled CPU JAX wheels by default; CUDA-enabled JAX still needs explicit installation (`jax[cuda13]`).
  Evidence: `uv pip install -e .` installed `jax==0.9.0` and `jaxlib==0.9.0` without CUDA.
- Observation: The BT4 weights file does not set `network_format.input_embedding`, but LC0’s loader upgrades multihead attention nets to `INPUT_EMBEDDING_PE_DENSE` during load.
  Evidence: The raw protobuf lacks `input_embedding`, yet LC0’s `loader.cc` sets PE_DENSE for multihead weights; the ONNX export uses dense positional embedding accordingly.
- Observation: The Lichess standard rated index currently lists `lichess_db_standard_rated_2025-12.pgn.zst` as the newest monthly dump (as of 2026-02-01), so tooling should always re-check the index rather than hard-coding a month.
  Evidence: The Lichess standard index page lists monthly dump filenames in chronological order, with 2025-12 as the latest entry.
- Observation: Sampling 200k rated Lichess games from the latest month produced only 7 games that met the classical 2400+ filter, indicating this slice is too sparse for the 1k target.
  Evidence: `tools/filter_pgn.py` reported `Kept games: 7` when filtering `data/lichess/lichess_sample_raw_200k.pgn`.
- Observation: Including rapid + classical at 2400+ still yields only a few dozen games from the 200k rated sample, so a larger scan or alternate sampling strategy is required to hit 1k.
  Evidence: The partial filtered output contains only 23 games so far (counted via `rg '^\\[Event'`).
  Update: The rapid+classical filter produced 23 games and ~3,975 FENs in `data/lichess/lichess_sample_2400_rapid_classical.*`.
- Observation: Lowering the rapid+classical threshold to 2000+ yields enough games for the 1k target (about 1,505 in the 200k sample).
  Evidence: A header-only scan of `data/lichess/lichess_sample_raw_200k.pgn` reports 1,505 games with min Elo >= 2000.
- Observation: Full 1,000-game FEN extraction yields ~80k positions; activation dumps are heavy, so a 5k-position subset is used for faster iteration.
  Evidence: `data/lichess/lichess_sample_2000_rapid_classical.fens` has ~80,566 lines; `data/lichess/lichess_sample_2000_rapid_classical_5k.fens` is the capped subset.
- Observation: The BT4 template `.pb.gz` contains an empty `policy_head_map` entry that must be pruned before serialization.
  Evidence: Export initially failed with missing required `policy_head_map` fields until empty entries were removed.
- Observation: LC0 training tar filenames sometimes use 4-digit times (HHMM), not 6-digit (HHMMSS), which required a more flexible parser.
  Evidence: The training data index lists entries like `training-run1--20200711-2017.tar`.
- Observation: LC0 does not accept a `--uci` flag; the binary should be launched directly for UCI sessions.
  Evidence: `lc0 --uci` exits with "Unknown command line flag".
- Observation: python-chess UCI sessions can time out on LC0 with the default timeout, so the Elo bench/UI need a higher UCI timeout.
  Evidence: `tools/elo_bench.py` hit `TimeoutError` mid-game until the timeout was raised.
- Observation: LC0 chunk-derived FENs always have `fullmove_number=1`, so `min_ply` filters drop everything; use phase/piece filters for self-play chunks.
  Evidence: Filtering `lc0_100k.fens` with `--min-ply 12` kept 0 positions, while phase/piece filtering kept 51,113 positions.
- Observation: The new audit shows the current static sparse separator and puzzle-tag pipeline is not yet a faithful Schut reproduction.
  Evidence: `REPO_AUDIT_AND_NEXT_STEPS.md` identifies missing dynamic MCTS rollout pairs, novelty filtering, teachability filtering, and unpooled activation storage.
- Observation: The first dynamic smoke run found that the broadcast 2400 classical FEN file on the VM was empty, while `data/runs/test_eval_pipeline/lichess/human.eval.fens` had usable test roots.
  Evidence: Re-running the smoke with 20 eval FEN roots kept 1 LC0 rollout pair, produced 6 trajectory FENs, materialized a `(1, 65536)` flat difference matrix, solved a CVXPY concept with status `optimal`, and wrote a novelty report.
- Observation: FEN-only trajectory dumps are not sufficient for BT4 dynamic concepts because continuation states need LC0 history planes.
  Evidence: PR review showed PV child encodings differ with and without history. The rollout builder now emits `trajectory.records.jsonl` with rolling `history_fens` and stable `activation_keys`, the materializer consumes those keys, and `gcp_dynamic_smoke_records_20260427` validated that corrected path with LC0.
- Observation: The direct flat sparse dynamic solve is now the scaling bottleneck. Solving 72 train rows over 65,536 flat features with CVXPY/SCS on `pipeline-vm` stayed active for about 30 minutes and was terminated after preserving the pair artifacts.
  Evidence: `gcp_dynamic_large_20260427_174945` materialized flat pairs with shape `(93, 65536)` and split them 72/21, but `tools/solve_dynamic_concepts.py --mode flat` did not finish in the practical validation window. Re-materializing the same rollout pairs with `--mode mean` produced `(93, 1024)` differences and completed solve/evaluation/baselines/prototypes/curriculum/policy-margin reporting successfully.
- Observation: The first larger mean-pooled causal patch run validated plumbing but not causal strength.
  Evidence: `dynamic_sparse_mean/policy_margin_report.json` reported `mean_delta_margin=-2.421438694000244e-08`, `top1_change_rate=0.0`, and `skipped_rows=0` on 16 held-out rows at `alpha=0.1`.
- Observation: Deterministic feature screening makes the first larger flat validation tractable, but causal patch effects are still tiny at the tested patch scale.
  Evidence: `gcp_dynamic_large_20260427_174945/concepts/dynamic_sparse_screened_2048` solved the `(72, 65536)` train split with a 2048-feature screen and status `optimal`, then reported held-out constraint satisfaction `0.762`, held-out margin satisfaction `0.190`, and policy-margin `mean_delta_margin=-4.4330954551696777e-07` at `alpha=0.1`.

## Decision Log


- Decision: Target the specific network `BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz` and make it the only required net for the first implementation.
  Rationale: This aligns with the user's request and ensures the oracle and Flax parity work is tightly scoped.
  Date/Author: 2026-01-29 / Codex
- Decision: Keep the repository structure minimal (single `lc0jax` package plus `tools`, `tests`, `data`, and `models`).
  Rationale: The codebase is expected to be small and the user asked to reduce folder sprawl.
  Date/Author: 2026-01-29 / Codex
- Decision: Defer all diffusion-related work entirely and remove it from the scope.
  Rationale: The user explicitly requested skipping diffusion for now.
  Date/Author: 2026-01-29 / Codex
- Decision: Use an ONNX oracle path as the primary regression guardrail and only accept "loose parity" for initial Flax outputs.
  Rationale: It de-risks correctness of encoding and model mapping while still enabling progress.
  Date/Author: 2026-01-29 / Codex
- Decision: Use the official LC0 binary and its bundled `leela2onnx` for ONNX export instead of a custom converter.
  Rationale: This matches the user's environment and avoids maintaining a forked exporter.
  Date/Author: 2026-01-29 / Codex
- Decision: Pin the LC0 release to the current latest official release, v0.32.1 (published 2025-11-23).
  Rationale: The user requested the latest official release; pinning the version keeps the pipeline reproducible.
  Date/Author: 2026-01-29 / Codex
- Decision: Accept PGN as the primary dataset input for both human and LC0 self-play data, with support for FEN lists as a lightweight intermediate format.
  Rationale: PGN is the standard distribution format, while FEN lists simplify iteration and caching.
  Date/Author: 2026-01-29 / Codex
- Decision: Target CUDA-enabled JAX as the default, while preserving a CPU-only fallback path. Use CUDA 13 wheels (`jax[cuda13]`) on Linux unless the environment requires the local CUDA toolchain (`jax[cuda13-local]`).
  Rationale: The primary environment is GPU-first and the JAX installation guidance prefers CUDA 13 wheels on Linux; CPU fallback preserves portability.
  Date/Author: 2026-01-29 / Codex
- Decision: Use `pandas` as the dataset dependency for PGN and metadata handling.
  Rationale: It is widely available, stable, and sufficient for the dataset sizes expected in this project.
  Date/Author: 2026-01-29 / Codex
- Decision: Generate the LC0 policy move list from LC0 v0.32.1 `encoder.cc` and store it as `lc0jax/policy_moves.txt`.
  Rationale: This preserves the exact policy index order used by LC0 without re-deriving a potentially mismatched mapping.
  Date/Author: 2026-01-29 / Codex
- Decision: Vendor the LC0 protobuf schemas (`net.proto`, `chunk.proto`) from lczero-common into `lc0jax/proto/`.
  Rationale: The repo needs a stable, local copy of the schema to parse `.pb.gz` weights, and the upstream LC0 repo does not include the `.proto` files directly.
  Date/Author: 2026-01-29 / Codex
- Decision: Use `uv` to create a local virtual environment and install build-time dependencies (grpcio-tools, python-chess) when system `venv`/`pip` is unavailable.
  Rationale: The base system lacks `python3-venv` and `pip`, but `uv` is present and allows isolated installs without sudo.
  Date/Author: 2026-01-29 / Codex
- Decision: Use CUDA-enabled JAX (`jax[cuda13]`) inside `.venv` and verify GPU availability via `jax.devices()` after install.
  Rationale: The primary target environment is GPU-first; validating the backend avoids silent CPU fallbacks.
  Date/Author: 2026-02-01 / Codex
- Decision: Human data will use Lichess standard rated games filtered to classical time controls with both players >= 2400 Elo.
  Rationale: The user requested the highest rating tier with classical time control; this provides a strong human baseline.
  Date/Author: 2026-02-01 / Codex
- Decision: Computer data will start from existing LC0 self-play training chunks; add a conversion script that emits one-ply PGNs/FENs per position.
  Rationale: LC0 publishes self-play chunks directly; chunk records are position-level, so PGN conversion is limited to one move.
  Date/Author: 2026-02-01 / Codex
- Decision: Organize code into `modeling`, `training`, `uci`, and `interpretability` subpackages, with compatibility wrappers at the top level.
  Rationale: The user wants model/training, UCI integration, and interpretability clearly separated while keeping the repo easy to navigate.
  Date/Author: 2026-02-01 / Codex
- Decision: Add `tools/download_data.py` to pull the latest Lichess standard PGNs and LC0 training tarballs, and always resolve the latest month from the online index instead of hard-coding it.
  Rationale: The user requested the latest month and a helper script; the index-driven approach keeps downloads current.
  Date/Author: 2026-02-01 / Codex
- Decision: Add the `zstandard` dependency to support decompression of `.zst` Lichess dumps.
  Rationale: The Lichess monthly PGNs are distributed as `.zst`, and downstream filtering expects plain PGN.
  Date/Author: 2026-02-01 / Codex
- Decision: Use the most recent Lichess standard dump at the time of implementation (`2025-12`) for the initial human dataset, but confirm via the index each time the script runs.
  Rationale: Ensures the dataset matches the latest available month while avoiding stale hard-coded filenames.
  Date/Author: 2026-02-01 / Codex
- Decision: Add a lightweight LC0 benchmark wrapper (`tools/lc0_benchmark.py`) to capture quick performance baselines alongside functional tests.
  Rationale: This mirrors how LC0 itself validates engine performance and provides an early signal when changes degrade throughput.
  Date/Author: 2026-02-01 / Codex
- Decision: Add a streaming Lichess sampler that filters games while downloading, to avoid pulling full monthly dumps when only ~1k games are needed.
  Rationale: The monthly dump is tens of GB compressed; streaming reduces disk and bandwidth requirements for early experiments.
  Date/Author: 2026-02-01 / Codex
- Decision: Use a 200k-game rated Lichess sample as a proxy for initial filtering, then reassess sampling strategy for the 2400+ classical target.
  Rationale: Directly scanning for 1k high-rated classical games in the full dump is too slow; a large sample gives an initial read on sparsity.
  Date/Author: 2026-02-01 / Codex
- Decision: Pull the latest LC0 training tar (`training-run2--20251215-1017.tar`) and aggregate multiple chunk files to reach 1,000 positions.
  Rationale: The individual chunk files are small; aggregating across chunks is required to reach the target sample size.
  Date/Author: 2026-02-01 / Codex
- Decision: Use a 2000+ rapid/classical filter to hit 1,000 human games in the sampled PGN.
  Rationale: 2200+ and 2100+ were insufficient; 2000+ yields >1,000 games while still remaining strong human play.
  Date/Author: 2026-02-01 / Codex
- Decision: Use a 5k-position subset for initial activation dumping to keep iteration time reasonable; scale up later if needed.
  Rationale: The full 80k-position FEN list is costly to process during early iterations.
  Date/Author: 2026-02-01 / Codex
- Decision: Use the Lichess broadcast API as the primary human OTB-style data source (download PGNs via `/api/broadcast/*.pgn`).
  Rationale: Broadcasts are official OTB events and provide a stronger human baseline than online games.
  Date/Author: 2026-02-01 / Codex
- Decision: Export BT4 weights using LINEAR16 encoding with a template `.pb.gz` to preserve LC0 metadata.
  Rationale: The target LC0 weights use LINEAR16, and the template contains required format fields for UCI compatibility.
  Date/Author: 2026-02-01 / Codex
- Decision: Download and pin the BT4 model artifact with SHA256 `e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618` (382,645,315 bytes).
  Rationale: Reproducibility requires a fixed, checksummed model file.
  Date/Author: 2026-01-29 / Codex
- Decision: Build LC0 v0.32.1 from source on Linux using Meson/Ninja (via `uv`), because no Linux binary is published in the official release assets.
  Rationale: The ONNX exporter depends on the official LC0 binary and `leela2onnx`; building from source is required on this platform.
  Date/Author: 2026-01-29 / Codex
- Decision: Use ONNX input/output shapes from the exported BT4 model: input `/input/planes` is `[batch, 112, 8, 8]`, policy output is `[batch, 1858]`, WDL output is `[batch, 3]`, and MLH is `[batch, 1]`.
  Rationale: These shapes define the encoder plane count and policy mapping size used across inference and masking.
  Date/Author: 2026-01-29 / Codex
- Decision: The BT4 network format specifies `INPUT_CLASSICAL_112_PLANE` (enum 1), so the default encoder format should be classical rather than canonical.
  Rationale: The ONNX model expects the classical 112-plane input format, and canonicalization would change the network inputs.
  Date/Author: 2026-01-29 / Codex
- Decision: Use the attention policy mapping table (`kAttnPolicyMap`) to map 67x64 attention logits to the 1858 move indices and store it as `lc0jax/policy_attn_map.txt`.
  Rationale: BT4 uses the attention policy head; the ONNX graph applies the same mapping table before producing policy logits, so parity requires the identical mapping.
  Date/Author: 2026-01-29 / Codex
- Decision: Apply LC0’s ONNX weight transposition convention (reshape to dims, then transpose with order `{1,0}`) when mapping `.pb.gz` weights to JAX matrices.
  Rationale: LC0 stores dense weights in output-major order; transposition is required to match ONNX matmul semantics and achieve parity.
  Date/Author: 2026-01-29 / Codex
- Decision: Use LC0’s residual scaling factor `alpha = (2 * num_encoder_blocks)^(-0.25)` for attention-body FFN and MHA residuals.
  Rationale: This matches the ONNX exporter’s residual scaling and is required for parity with BT4.
  Date/Author: 2026-01-29 / Codex
- Decision: Proceed with the currently collected broadcast dataset (36,039 FENs from broadcasts) instead of re-downloading to reach the 10k-game target.
  Rationale: The user approved using the current games and continuing the pipeline without re-downloading.
  Date/Author: 2026-02-04 / Codex
- Decision: Add optional FEN filtering (ply/phase/piece count) to mitigate overly-similar opening positions, but keep it disabled by default to preserve reproducibility.
  Rationale: Opening positions cluster too tightly; filtering is useful for interpretability while preserving backwards compatibility.
  Date/Author: 2026-02-04 / Codex
- Decision: Treat puzzle positions as a separate human dataset (default rating >= 2500) and keep the broadcast dataset unchanged.
  Rationale: Puzzles are tactical and distributionally distinct from full games, so they should be analyzed separately.
  Date/Author: 2026-02-04 / Codex
- Decision: Add a “policy disagreement” filter (LC0 search vs raw policy) to better align with Schut-style novelty filtering.
  Rationale: The paper emphasizes selecting positions where two policies disagree; LC0 search vs no-search is a practical proxy.
  Date/Author: 2026-02-04 / Codex
- Decision: Keep mean-pooled activation dumps as the backwards-compatible default, but add `--activation-mode flat` and optional raw token storage for Schut-faithful square-aware experiments.
  Rationale: Existing tools and data expect 2D embeddings, while dynamic concepts need spatial information that mean pooling discards.
  Date/Author: 2026-04-27 / Codex
- Decision: Treat dynamic rollout extraction, large activation dumps, large SVD novelty runs, and teachability training as GCP workloads.
  Rationale: These steps are compute-heavy and should not be run on the local workspace except as fixture-sized smoke tests.
  Date/Author: 2026-04-27 / Codex
- Decision: Preserve flat dynamic pair artifacts but use a mean-pooled fallback to complete the first larger end-to-end validation when the direct flat CVXPY/SCS solve exceeded the practical runtime window.
  Rationale: The larger run still needed a complete train/test/evaluation/prototype/curriculum/policy-margin artifact, and the stalled flat solve exposed a concrete implementation bottleneck for the next cycle.
  Date/Author: 2026-04-27 / Codex
- Decision: Add deterministic feature screening as the first scalable flat solver path while keeping the exact unscreened sparse CVXPY objective available.
  Rationale: The Schut-style L1 objective remains unchanged on the selected support, outputs retain the original feature dimension for evaluation and patching, and selected feature indices/scores are stored for reproducibility.
  Date/Author: 2026-04-27 / Codex

## Outcomes & Retrospective


- The LC0/JAX substrate and static concept baseline are implemented, but the repository is not yet Schut-faithful. As of 2026-04-27, the initial parity infrastructure now includes square-aware activation modes, a reusable sparse paired-difference solver, screened flat sparse solves, LC0 MCTS rollout pairs, history-aware trajectory activations, root-grouped held-out splits, held-out direction evaluation, prototype/control selection, curriculum export, dynamic report cards, baselines, policy-margin patch reports, and the machine-vs-human SVD novelty metric. A larger GCP validation has now produced 40 LC0 rollout-pair records, a complete mean-pooled held-out report, and a complete screened-flat held-out report; the immediate technical blockers are feature-screening scale comparisons, stronger causal patch calibration, and teachability filtering.

## Context and Orientation


This repository currently contains `ExecPLANS.md` and this plan file. The project will be a Python package named `lc0jax` that implements LC0 inference and a Schut-style concept discovery workflow for chess. LC0 (Leela Chess Zero) is a chess engine that uses a neural network; the network weights are stored in `.pb.gz` protobuf files. The target network is a BT4 family model with a transformer-like architecture; "BT4-1024x15x32h" indicates 1024 channels, 15 blocks, and 32 attention heads. The plan includes an ONNX "oracle" path that converts the `.pb.gz` weights to an ONNX model using the official LC0 binary and its bundled `leela2onnx`, then runs the ONNX model with onnxruntime to validate correctness. Policy output refers to the move logits; value/WDL output refers to the win/draw/loss head (or scalar value) produced by LC0 networks. Input "planes" are fixed-format 8x8 feature maps derived from a chess position and a short move history. "Activation patching" means adding or subtracting a learned concept direction to an intermediate activation and measuring the change in the model's output. PGN is used as the primary dataset format, with FEN lists as an intermediate cache to speed repeated experiments.

The current environment includes an NVIDIA GeForce GTX 1660 Ti, driver 581.80, and CUDA 13.0. The plan defaults to the CUDA 13 pip wheels but keeps CPU-only fallbacks for portability and for cases where the CUDA 13 wheel cannot be used.

The repository layout stays minimal but is now structured into focused subpackages. The root contains `pyproject.toml`, `README.md`, `PLANS.md`, and `AGENTS.md`. The `lc0jax/` package is organized into `modeling/` (encoder, weights, model, inference), `training/` (training utilities and chunk parsing), `uci/` (LC0 interop such as the ONNX oracle and export hooks), and `interpretability/` (datasets, activations, concept discovery). Compatibility wrappers remain at `lc0jax/*.py` for older imports. The `tools/` directory contains CLI scripts (export ONNX, run oracle, compare oracle vs Flax, dump activations, convert chunks, run concept discovery). The `tests/` directory holds unit and integration tests, `data/` holds small FEN sets and sample outputs, and `models/` holds the BT4 `.pb.gz` file and its derived ONNX export.

All terms used in the plan are defined here. "Oracle" means a trusted baseline produced by onnxruntime. "Planes" are the LC0-style input tensors. "Policy mapping" means converting between network logits and chess moves. "BT4" refers to the specific transformer-style LC0 network family that this plan targets.

## Plan of Work


This plan is organized into four milestones, each providing a user-visible capability and an explicit verification path. Each milestone is independently verifiable and builds toward a full, open-source reproduction of the Schut paper methodology for LC0. If a milestone requires a prototype or spike, it is described explicitly as such and either promoted or discarded based on concrete, observable results.

### Milestone 1: Bootstrap + ONNX oracle path


At the end of this milestone, the repo has a working package skeleton, dependencies installed, the BT4 `.pb.gz` model stored under `models/`, and an ONNX export plus a runner that can execute the ONNX model for a small FEN list. The ONNX export must use the official LC0 binary and its bundled `leela2onnx` (not a custom exporter). The visible behavior is that `python tools/run_oracle.py --onnx models/BT4.onnx --fens data/fens.txt` prints policy/value outputs without errors. This milestone includes a prototype step to validate the LC0 binary + `leela2onnx` pairing; if the bundled exporter fails, the plan requires retrying with a different official LC0 release and recording the exact version used.

### Milestone 2: LC0 encoder, policy mapping, and legality masking


At the end of this milestone, a `python-chess` based encoder produces LC0 input planes that match the BT4 network input shape, a policy mapping exists for the actual output shape produced by the ONNX model, and legality masking ensures only legal moves are selected. Tests cover deterministic encoding, side-to-move and castling changes, en passant, and policy legality. The visible behavior is that the encoder tests pass and the oracle runner can accept encoded planes and produce outputs for all FENs without NaNs, while masked argmax always yields a legal move.

### Milestone 3: Load .pb.gz weights and implement the BT4 Flax model


At the end of this milestone, the BT4 `.pb.gz` file can be parsed into named tensors, the Flax model can be constructed with these weights, and a forward pass produces policy/value outputs with loose parity against the ONNX oracle (top-5 overlap and value correlation over the test FENs). This milestone includes an ONNX graph inspection step to identify the BT4 block structure and ensure that the Flax module matches the ONNX computation graph. The visible behavior is that `python tools/compare_oracle_flax.py` reports the agreed parity metrics on the test FEN set.

For export back into LC0, add a follow-on step that writes the JAX parameter tree into the LC0 protobuf `Weights` format (including proper encodings and network format fields), then compresses to `.pb.gz`. As an alternative, consider exporting to ONNX and using `lc0 onnx2leela` to generate `.pb.gz`, but only if the ONNX path preserves the exact input/output formats.

### Milestone 4: Activation capture and Schut-style concept discovery


At the end of this milestone, the model can dump activation embeddings for large datasets sourced from PGN files (human and LC0 self-play), and it can also ingest precomputed FEN lists as a lightweight intermediate. The concept discovery pipeline computes distribution-shift directions between the two datasets, and activation patching produces measurable changes in policy/value outputs. The visible behavior is that `python tools/dump_activations.py` produces shards of embeddings for both datasets, `python tools/discover_concepts.py` writes a report with top concepts and prototypes, and `python tools/discover_concepts.py --patch` (or a separate tool) shows statistically significant shifts for at least five concept directions.

### Milestone 5: Export trained JAX weights back to LC0 (.pb.gz)

At the end of this milestone, a locally fine-tuned JAX BT4 model can be exported to a `.pb.gz` weights file that LC0’s UCI engine can load. The visible behavior is that `lc0 describenet --weights exported.pb.gz` works, and a small oracle comparison confirms parity against the JAX model on a test FEN set.

## Concrete Steps


All commands below are intended to run from the repository root `/home/ubuntu/schutpaper` unless noted.

First, create the minimal package skeleton and dependencies. This plan assumes a standard Python virtual environment and editable install. The expected outcome is that `python -c "import lc0jax"` succeeds. GPU-first means installing CUDA-enabled JAX; the CPU fallback is the plain `jax` and `jaxlib` packages. Use the CUDA wheel that matches your local CUDA runtime and record the exact install command in the Decision Log.

    python -m venv .venv
    . .venv/bin/activate
    pip install -U pip

    # GPU-first: install CUDA-enabled JAX. For Linux with CUDA 13, the JAX docs recommend:
    pip install -U "jax[cuda13]"

    # If you must use a locally installed CUDA toolkit instead of the pip-provided CUDA libraries:
    # pip install -U "jax[cuda13-local]"

    # CPU fallback (only if CUDA is not available):
    # pip install -U jax jaxlib

    pip install -e .

If the system Python lacks `venv` support, use `uv` to create a venv and install dependencies (this is the current environment). The expected outcome is a working `.venv/bin/python` and a successful editable install.

    uv venv .venv
    uv pip install --python .venv/bin/python numpy python-chess protobuf onnx onnxruntime grpcio-tools pytest tqdm rich pandas zstandard
    uv pip install --python .venv/bin/python -e .

Install Meson and Ninja in the same venv to build LC0 from source on Linux (no official Linux binary is available for v0.32.1).

    uv pip install --python .venv/bin/python meson ninja
    git clone --depth 1 --branch v0.32.1 https://github.com/LeelaChessZero/lc0 /tmp/lc0-src
    PATH=/home/ubuntu/schutpaper/.venv/bin:$PATH /tmp/lc0-src/build.sh release
    /tmp/lc0-src/build/release/lc0 --help | head -3

Download the official LC0 release archive that includes the `lc0` binary and bundled `leela2onnx`, then place both binaries on your PATH or record their absolute paths for use by the tools. Record the LC0 release version and archive SHA256 in the Decision Log. Then download the target BT4 model into `models/` and record its file size and SHA256 to preserve reproducibility.

For Linux, if the LC0 release assets do not include a prebuilt binary for your platform, build from the official v0.32.1 source release and record the git tag used.

    mkdir -p models
    curl -L -o models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz https://storage.lczero.org/files/networks-contrib/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz
    sha256sum models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz

Export to ONNX using the official LC0 binary and its bundled `leela2onnx`, then run the oracle. The expected outcome is printed policy and value outputs for each FEN.

    python tools/export_onnx.py --lc0 /tmp/lc0-src/build/release/lc0 --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --onnx models/BT4.onnx
    python tools/run_oracle.py --onnx models/BT4.onnx --fens data/fens.txt

Run tests as they are added. The expected outcome is that pytest reports all tests passing and indicates the specific test names listed in this plan.

    pytest -q

Capture a quick LC0 performance baseline using the engine's benchmark modes.

    python tools/lc0_benchmark.py --lc0 /tmp/lc0-src/build/release/lc0 --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --mode benchmark

Generate protobuf bindings when a `protoc` compiler is available (required before parsing `.pb.gz` weights).

    python tools/gen_proto.py --protoc protoc --out lc0jax

Prepare dataset inputs for the concept pipeline by converting PGN to FEN lists, then run activation dumps. The expected outcome is that the FEN list files are created and activation shards are written under a timestamped output directory.

    python tools/pgn_to_fens.py --pgn data/human.pgn --out data/human.fens
    python tools/pgn_to_fens.py --pgn data/lc0.pgn --out data/lc0.fens
    python tools/dump_activations.py --fens data/human.fens --out data/activations/human
    python tools/dump_activations.py --fens data/lc0.fens --out data/activations/lc0

For human datasets, prefer the Lichess monthly standard rated PGN dumps (download list and checksums live at https://database.lichess.org/standard/). Use `python tools/download_data.py lichess --out-dir data/lichess --decompress` to fetch the latest month and decompress to PGN before filtering. For smaller, targeted human samples, the Lichess API provides a user game export endpoint (see https://lichess.org/forum/general-chess-discussion/exporting-pgns for an example). For computer data, prefer locally generated LC0 self-play or rollouts from our BT4 model to avoid mixing bot/human pools.

When filtering human data, apply a high rating threshold (to be defined) and restrict to classical time controls using the Lichess time-control formula (estimated duration = base + 40 * increment; classical if >= 1500 seconds). Use `tools/filter_pgn.py` to derive a filtered PGN and corresponding FEN list before activation dumps.

For LC0 self-play training chunks, use `python tools/download_data.py lc0-chunks --out-dir data/lc0-training --count 1 --min-size 1000000` to fetch the newest tarball, then convert it with `tools/chunks_to_pgn.py`. Chunk records are position-level, so the PGN output should be treated as single-move games, not full game transcripts.

## Validation and Acceptance


Acceptance is based on observable behavior and test outputs, not just code structure. The implementation is accepted when all of the following are true:

The package installs and imports. Running `python -c "import lc0jax; print(lc0jax.__version__)"` prints a version string without error. Running `python -c "import jax; print(jax.devices())"` shows a CUDA device when available; if no GPU is present, it should list a CPU device and still run.

The LC0 binary is verified. Running `lc0 --help` prints `v0.32.1` in the banner.

The ONNX oracle path succeeds. Running `python tools/run_oracle.py --onnx models/BT4.onnx --fens data/fens.txt` prints a policy tensor and a value/WDL tensor for each FEN, and no NaNs appear in the output.

The encoder is correct enough for parity. `pytest -q` reports the encoder tests (`tests/test_encoder.py`) and legality tests (`tests/test_policy_legality.py`) passing. The tests must include: determinism on repeated FEN encoding, castling rights plane sensitivity, en passant encoding, and side-to-move symmetry.

Flax parity is "loose but correct." Running `python tools/compare_oracle_flax.py --onnx models/BT4.onnx --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --fens data/fens.txt` reports top-5 policy overlap >= 0.80 and a value/WDL correlation >= 0.90 across the FEN suite.

Schut-style concepts are reproducible. Running `python tools/dump_activations.py` on two datasets (PGN inputs or cached FEN lists) produces embedding files, and running `python tools/discover_concepts.py` produces a report listing at least 20 concepts with prototype FENs. A patching run demonstrates statistically significant shifts for at least five concept directions in either policy logits or value/WDL.

## Idempotence and Recovery


All steps are designed to be rerun safely. Re-running the model download should overwrite the file in `models/` without changing outputs; record the SHA256 to detect accidental changes. Re-running ONNX export overwrites `models/BT4.onnx`; this is safe as long as the exporter is deterministic. If ONNX export fails, the recovery path is to re-run the converter in isolation and validate that onnxruntime can load the resulting file, or try a different official LC0 release that bundles a compatible `leela2onnx`. For dataset and activation dumps, use timestamped output directories under `data/` to avoid accidental overwrites, and provide a `--resume` or `--skip-existing` flag in the tools to allow safe retries. PGN-to-FEN conversion should be cached so repeated runs do not re-parse the same PGN unless inputs change.

## Artifacts and Notes


Keep minimal, human-readable artifacts that prove success. This includes the small `data/fens.txt` test set, a short `data/oracle_sample.txt` output captured from `tools/run_oracle.py`, and a `data/concept_report.md` generated by the concept discovery tool. For dataset caching, also keep a small derived FEN list for each PGN input (for example `data/human.fens` and `data/lc0.fens`). These artifacts should be small and are intended to provide quick confidence checks for a novice.

## Interfaces and Dependencies


The implementation is expected to use Python 3.10+ with the following dependencies: `jax`, `jaxlib`, `flax`, `numpy`, `scipy`, `python-chess`, `protobuf`, `onnx`, `onnxruntime`, `tqdm`, and `rich`. JAX should be installed with CUDA support when available; CPU-only installs must still function. Optional dependencies for dataset handling include `pandas` or `polars`; choose one and record the decision.

The public module interfaces below must exist at the end of Milestone 3 and remain stable. They should be implemented as plain Python functions and Flax modules with clear docstrings.

In `lc0jax/encode.py`, define:

    def encode_board(board, history, *, planes_layout: str) -> np.ndarray:
        """Return LC0 input planes as float32, shape [C, 8, 8]."""

In `lc0jax/policy.py`, define:

    def move_to_policy_index(move, policy_format: str) -> int:
        """Map a chess move to a policy index for the current output format."""

    def policy_index_to_move(index, policy_format: str):
        """Map a policy index back to a chess move."""

    def legal_move_mask(board, policy_format: str) -> np.ndarray:
        """Return a boolean mask over the policy indices."""

In `lc0jax/weights.py`, define:

    @dataclass
    class WeightsBundle:
        tensors: dict
        metadata: dict

    def load_pb_gz(path: str) -> WeightsBundle:
        """Parse .pb.gz into named tensors and metadata."""

In `lc0jax/model.py`, define:

    class Bt4Model(nn.Module):
        """Flax module mirroring the BT4 ONNX graph."""
        def __call__(self, planes, *, capture: bool = False):
            """Return (policy_logits, value_or_wdl, activations_or_none)."""

In `lc0jax/inference.py`, define:

    def forward(params, planes, *, mask=None, capture: bool = False):
        """Run Flax forward pass and apply legality masking if provided."""

In `lc0jax/oracle.py`, define:

    def run_onnx(onnx_path: str, planes: np.ndarray) -> dict:
        """Return ONNX outputs as numpy arrays with stable keys."""

In `lc0jax/activations.py`, define:

    def dump_activations(params, dataset_iter, *, out_dir: str, layer: str):
        """Write activation embeddings and metadata shards to disk."""

In `lc0jax/datasets.py`, define:

    def pgn_to_fens(pgn_path: str, *, out_path: str, max_positions: int | None = None, ply_stride: int = 1) -> list[str]:
        """Parse PGN into a list of FENs and optionally write them to disk."""

    def iter_fens(path: str):
        """Yield FEN strings from a newline-delimited FEN file."""

In `lc0jax/concepts.py`, define:

    def discover_concepts(embeddings_a, embeddings_b, *, method: str):
        """Return concept directions and scores."""

    def patch_activations(params, sample, concept_vec, *, alpha: float):
        """Return output deltas after activation patching."""

## References (for later lookup only)


These references are recorded for future lookups but are not required to execute the plan because all essential instructions are embedded above. The Schut paper is at https://arxiv.org/abs/2310.16410. The target LC0 network file is at https://storage.lczero.org/files/networks-contrib/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz. The LC0 source repository (encoder, policy map, protobuf definitions, ONNX exporter) is https://github.com/LeelaChessZero/lc0, the LC0 project site is https://lczero.org, and the official LC0 releases page for binaries is https://github.com/LeelaChessZero/lc0/releases. JAX installation guidance is at https://jax.readthedocs.io/en/latest/installation.html and the CUDA wheel index is https://storage.googleapis.com/jax-releases/jax_cuda_releases.html.

Note: Plan created on 2026-01-29 as the initial baseline in response to the user's request to target the BT4 network, simplify the repo layout, and omit diffusion-related work. Revised on 2026-01-29 to replace non-ASCII quotes and convert layout/reference lists into prose to comply with ExecPlan formatting guidance. Revised on 2026-01-29 to require the official LC0 binary with bundled `leela2onnx`, PGN-first datasets with FEN caching, and GPU-first JAX with CPU fallback. Revised on 2026-01-29 to pin LC0 v0.32.1, update CUDA 13 JAX install guidance, and record the detected CUDA 13.0 environment.
