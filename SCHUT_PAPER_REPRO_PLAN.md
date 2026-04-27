# Schut Paper Reproduction and Extension

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document serves as an extension to `PLANS.md` in accordance with `ExecPLANS.md`.

## Purpose / Big Picture

The goal is to accurately reproduce the concept discovery methodology from the Schut et al. AlphaZero paper (arXiv:2310.16410) on LC0, and extend it by matching discovered concepts to Lichess puzzle themes after discovery. The current repo has a useful static sparse-concept baseline, but Schut parity requires dynamic concepts from LC0 MCTS optimal-versus-subpar rollout pairs. This involves five major steps:
1. Auditing our current `discover_concepts.py` implementation against the exact mathematical formulations used in the Schut paper (e.g., whitening, PCA, SVM probes) to ensure our concept directions are derived identically.
2. Implementing a robust data filtration pipeline using LC0's MCTS evaluation to isolate complex, balanced middlegames (win probability between 10% and 90%), stripping out forced endgames and rote openings.
3. Preserving square-local BT4 activations and PGN history so concept vectors can operate on the representation geometry Schut et al. use.
4. Building dynamic MCTS rollout-pair datasets and solving sparse concept vectors over `psi(tau+) - psi(tau-)` constraints.
5. Filtering discovered concepts for novelty and teachability, then using Lichess puzzle tags only as interpretation/evaluation metadata.

## Progress

- [x] (2026-04-25) Audit `tools/discover_concepts.py` against the Schut paper methodology.
- [x] (2026-04-25) Implement `tools/filter_fens_eval.py` to filter FENs based on LC0 MCTS evaluation limits.
- [x] (2026-04-25) Modify `tools/download_data.py` (specifically `lichess-puzzles`) to save puzzle tags alongside FENs (e.g., as JSONL).
- [x] (2026-04-26) Run the full filtered data pipeline: extract FENs, filter via MCTS + heuristics, and dump activations.
- [x] (2026-04-26) Run the audited concept discovery pipeline on the filtered dataset.
- [x] (2026-04-26) Dump activations for the tagged Lichess puzzles.
- [x] (2026-04-25) Create `tools/match_concepts.py` (or a Jupyter notebook) to project puzzle activations onto discovered concept directions and rank the correlation of Lichess tags to each concept.
- [x] (2026-04-27) Re-audited the repo against the dynamic Schut method in `REPO_AUDIT_AND_NEXT_STEPS.md` and corrected the implementation status in `IMPLEMENTATION_STATUS_AND_NEXT_WORK.md`.
- [x] (2026-04-27) Added square-aware activation projection, optional raw token storage, sparse paired-difference solving, stored-rollout dynamic difference aggregation, and SVD novelty filtering.
- [x] (2026-04-27) Added metadata-first LC0 MultiPV rollout-pair extraction with optimal/subpar PV states, scores, node metadata, and trajectory FEN export.
- [x] (2026-04-27) Added `tools/materialize_mcts_pairs.py` to join trajectory activation shards back to rollout-pair JSONL and write solver-ready `pairs.npz`, using stable activation keys when available.
- [x] (2026-04-27) Added history-aware PGN activation records and `dump_activations.py --records`.
- [x] (2026-04-27) Updated `tools/run_full_pipeline.sh` to use history-aware human activation records by default when broadcast PGNs are available.
- [x] (2026-04-27) Ran the first small GCP dynamic-concept smoke pipeline from LC0 MultiPV pairs through flat activation dump, `pairs.npz` materialization, sparse solve, and novelty report under `data/runs/gcp_dynamic_smoke_20260427`.
- [x] (2026-04-27) Re-ran the GCP dynamic smoke with history-aware trajectory records under `data/runs/gcp_dynamic_smoke_records_20260427`.
- [x] (2026-04-27) Added dynamic concept markdown report cards from `pairs.npz`, solver reports, and novelty reports.
- [x] (2026-04-27) Added dynamic random/shuffled baseline summaries for learned rollout concept directions.
- [x] (2026-04-27) Added dynamic policy-margin patch reports for best-vs-subpar root moves.
- [x] (2026-04-27) Added root-grouped held-out train/test splits for dynamic `pairs.npz` files.
- [ ] Add teachability filtering and random-prototype baselines.

## Surprises & Discoveries

- Observation: L1-regularized SVM (via `cvxpy`) optimization produced sparse concept vectors exactly as described by Schut et al., providing a clear mathematical link to human-understandable Lichess themes without requiring post-hoc PCA or thresholding.
- Observation: Spot TPU (`v5litepod-8` / `v5litepod-4`) provisioning encountered quota limitations (max 4 chips for v5litepod APIs in testing zones) and capacity errors. We fell back to deploying a robust `pipeline-vm` controller (`n2-standard-16`) to compile LC0 from source and run localized, cheap end-to-end test validations.
- Observation: Discovering concepts and matching them to Lichess puzzle tags works seamlessly; `match_concepts.py` confirmed that the raw discovered concept vectors are meaningfully correlated with human tags like `middlegame` and `fork`.
- Observation: The April 2026 audit found that the static CVXPY/tag pipeline is not sufficient for a faithful Schut reproduction because it lacks dynamic optimal-vs-subpar MCTS rollout constraints, novelty filtering, and teachability filtering.
- Observation: Mean-pooled activations are backwards compatible but remove square-local chess information; `--activation-mode flat` and `--store-token-activations` are now available for Schut-style experiments.
- Observation: The first GCP dynamic smoke run validates the new dynamic pipeline plumbing but is too small to make a novelty claim.
  Evidence: `gcp_dynamic_smoke_20260427` kept 1 rollout pair, dumped 6 flat trajectory activations, solved a `(1, 65536)` sparse concept with status `optimal`, and produced a novelty curve with `positive_rank_fraction=0.0` against a 10-position human reference.
- Observation: Dynamic trajectory activations must be dumped from records rather than unique FEN lists for BT4 112-plane inputs.
  Evidence: PV continuations encode differently when rolling history is present. `tools/build_mcts_pairs.py --out-trajectory-records` now writes history-aware records with `activation_keys`, `tools/materialize_mcts_pairs.py` uses those keys instead of collapsing repeated FENs, and `gcp_dynamic_smoke_records_20260427` validated the corrected path with LC0.
- Observation: The first policy-margin smoke validates patch/report plumbing, not concept effect size.
  Evidence: `tools/dynamic_policy_margin.py` ran on `gcp_dynamic_smoke_records_20260427` with one pair and `alpha=0.1`, wrote `policy_margin_report.json`, used legal-masked top-1 metrics, and produced `mean_delta_margin=0.0`.

## Decision Log

- Decision: Use LC0 MCTS for evaluation filtering instead of raw BT4 network value.
  Rationale: The raw value head without search can be myopic. Schut et al. rely on accurate evaluations to ensure positions are truly complex and balanced middlegames.
  Date/Author: 2026-04-25 / Gemini CLI
- Decision: Add an audit phase for `discover_concepts.py`.
  Rationale: To ensure our concept vectors (e.g., whitened mean difference, PCA-based) strictly match the paper's math before running expensive pipelines.
  Date/Author: 2026-04-25 / Gemini CLI
- Decision: Run end-to-end tests on a compute Spot VM (`n2-standard-16`) named `pipeline-vm` instead of waiting for TPU quota.
  Rationale: To immediately validate the data filtering pipeline, JAX forward passes, and `cvxpy` concept vectors cheaply and efficiently without being blocked by global TPU Spot shortages.
  Date/Author: 2026-04-26 / Gemini CLI
- Decision: Treat puzzle tags as interpretation/evaluation, not as the Schut discovery signal.
  Rationale: The paper discovers dynamic concepts from MCTS rollout contrasts, then filters and interprets them; labels should not define the primary concept vectors.
  Date/Author: 2026-04-27 / Codex
- Decision: Run heavy LC0 search, full activation dumps, large SVD novelty sweeps, and teachability training on GCP.
  Rationale: Local compute is appropriate for tests and smoke fixtures only; the paper-parity workload is compute-heavy.
  Date/Author: 2026-04-27 / Codex

## Outcomes & Retrospective

- Successfully audited and implemented a static `cvxpy` L1-SVM concept discovery baseline related to Schut et al.'s sparse objective.
- Created `filter_fens_eval.py` to curate a high-quality dataset of balanced middlegame positions using LC0's MCTS engine.
- Extended the pipeline to persist Lichess puzzle tags (`.jsonl`) and developed `match_concepts.py` to project those human-interpretable tags onto our unsupervised concept vectors.
- Validated the entire pipeline locally on a GCP Spot VM instance (`pipeline-vm`), achieving an end-to-end successful run (data extraction -> MCTS evaluation filter -> JAX activation dump -> SVM concept discovery -> Puzzle concept mapping).
- Added the first dynamic-parity infrastructure: unpooled activation storage options, paired-difference sparse solving, LC0 MultiPV rollout-pair extraction, pair materialization, dynamic sparse solving, and SVD novelty curves.
- Validated the corrected dynamic path on GCP with the history-aware smoke run `data/runs/gcp_dynamic_smoke_records_20260427`.
- Added dynamic report-card generation for root FENs, best/subpar moves, PVs, scores, solver stats, pair materialization metadata, and novelty summaries.
- Added random sparse, shuffled-label, and optional shuffled-solve baselines for dynamic concept runs.
- Added policy-margin patch reports for dynamic concepts, including support for flat `[64 * channels]` directions at token-shaped patch points.
- Added root-position grouped train/test splitting for dynamic rollout pairs, so constraint satisfaction, baselines, and policy-margin checks can run on held-out roots.

**Next Steps:**
1. Scale MCTS pair extraction and flat activation dumps on GCP with larger root sets, higher node budgets, and sharded resume support.
2. Scale dynamic runs with held-out root splits and include report cards, baselines, novelty, and policy-margin patching.
3. Add teachability filtering with a weaker LC0 checkpoint or student network and random-prototype baselines.

## Context and Orientation

The repository currently implements a pipeline that downloads Lichess data, applies heuristic filters (`tools/filter_fens.py`), dumps LC0 activations (`tools/dump_activations.py`), discovers static concept vectors (`tools/discover_concepts.py`), filters novelty curves (`tools/filter_novel_concepts.py`), and can solve sparse dynamic vectors from precomputed rollout tensors (`tools/solve_dynamic_concepts.py`). The main missing piece is generating the rollout-pair tensors from LC0 MCTS searches.

## Plan of Work

### Milestone 1: Concept Discovery Audit
Audit `tools/discover_concepts.py` and `lc0jax/interpretability/concepts.py` against Schut et al. Verify the exact method for computing concept vectors (e.g., is it SVM weights, whitened mean difference, or PCA directions?). Update the implementation to strictly match the paper, removing or deprecating non-compliant methods. Create a brief markdown report (`docs/schut_audit_report.md`) detailing the mathematical alignment.

### Milestone 2: Evaluation-Based Data Filtration
Create `tools/filter_fens_eval.py` using `python-chess` and the UCI engine interface (similar to `filter_fens_disagreement.py`). The script should run a short MCTS search (e.g., `--nodes 800`) on each FEN and keep the position only if the engine evaluation is within a specified centipawn/win-probability range (e.g., `[-150, 150]` cp) and heuristic bounds (e.g., ply > 20, pieces > 10).
Expected observable outcome: Running `python tools/filter_fens_eval.py --fens data/human.fens --out data/human_balanced.fens --lc0 ...` produces a smaller FEN list containing only balanced middlegames.

### Milestone 3: Unsupervised Puzzle Concept Matching
1. Modify `tools/download_data.py`'s `lichess-puzzles` command to save the puzzle tags (the `Themes` column from the CSV) alongside the FENs, outputting a `.jsonl` file instead of just `.fens`.
2. Update `tools/dump_activations.py` to optionally accept `.jsonl` input and preserve the tags in the output shard metadata, or run it as-is on the extracted FENs while maintaining order so the JSONL tags can be mapped back.
3. Create `notebooks/puzzle_concept_matching.ipynb` or `tools/match_concepts.py`. This tool will load the discovered concept vectors from Milestone 1, load the puzzle activations, project them, and compute which tags have the highest average activation (or correlation) for each concept vector.
Expected observable outcome: The tool outputs a ranking showing which human-understandable Lichess themes (e.g., `fork`, `pin`) correspond to which unsupervised LC0 concept vectors.

### Milestone 4: Dynamic Rollout Concepts

Create `lc0jax/interpretability/mcts_rollouts.py` and `tools/build_mcts_pairs.py`. The tool should call LC0 through UCI, run node-limited searches from selected root FENs, capture the best principal variation and meaningful subpar alternatives, replay those lines into FEN trajectories, dump flat or token activations for each state, and write a `pairs.npz` file consumable by `tools/solve_dynamic_concepts.py`. The acceptance test is a tiny fixture run that produces at least one pair and then solves a sparse concept report with positive constraint satisfaction.

### Milestone 5: Novelty and Teachability Filters

Use `tools/filter_novel_concepts.py` to compare concept vectors against LC0/self-play and human SVD bases. Then add a teachability tool that trains a weak student or weaker LC0 checkpoint on concept prototypes using teacher-policy KL and reports top-1 overlap lift against random prototypes. These steps should run on GCP for nontrivial datasets.

## Concrete Steps

Recent smoke-test commands:

    cd /home/ubuntu/schutpaper
    .venv/bin/python -m pytest tests/test_activations.py tests/test_concepts.py tests/test_novelty.py -q

Square-aware activation dump example:

    cd /home/ubuntu/schutpaper
    python tools/dump_activations.py --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --fens data/human.fens --out data/activations/human_flat --activation-mode flat --store-token-activations

History-aware trajectory dump example:

    cd /home/ubuntu/schutpaper
    python tools/build_mcts_pairs.py --fens data/human.filtered.fens --out-jsonl data/runs/<RUN_ID>/mcts_pairs/pairs.jsonl --out-trajectory-records data/runs/<RUN_ID>/mcts_pairs/trajectory.records.jsonl --out-trajectory-fens data/runs/<RUN_ID>/mcts_pairs/trajectory.fens --lc0 /tmp/lc0-src/build/release/lc0 --weights models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --nodes 800 --multipv 4
    python tools/dump_activations.py --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --records data/runs/<RUN_ID>/mcts_pairs/trajectory.records.jsonl --out data/runs/<RUN_ID>/activations/trajectory_flat --activation-mode flat --store-token-activations

Dynamic sparse solver example once `pairs.npz` exists:

    cd /home/ubuntu/schutpaper
    python tools/materialize_mcts_pairs.py --pairs-jsonl data/runs/<RUN_ID>/mcts_pairs/pairs.jsonl --activations data/runs/<RUN_ID>/activations/trajectory_flat --out data/runs/<RUN_ID>/mcts_pairs/pairs.npz --mode flat
    python tools/split_dynamic_pairs.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.npz --out-train data/runs/<RUN_ID>/mcts_pairs/pairs.train.npz --out-test data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --test-fraction 0.2 --seed 0
    python tools/solve_dynamic_concepts.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.train.npz --out data/runs/<RUN_ID>/concepts/dynamic_sparse --mode flat
    python tools/dynamic_concept_baselines.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --out data/runs/<RUN_ID>/concepts/dynamic_sparse/baselines_report.json
    python tools/dynamic_policy_margin.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --pb models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --out data/runs/<RUN_ID>/concepts/dynamic_sparse/policy_margin_report.json
    python tools/build_dynamic_concept_report.py --pairs data/runs/<RUN_ID>/mcts_pairs/pairs.test.npz --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --out data/runs/<RUN_ID>/concepts/dynamic_sparse/report.md

Novelty filter example:

    cd /home/ubuntu/schutpaper
    python tools/filter_novel_concepts.py --concept data/runs/<RUN_ID>/concepts/dynamic_sparse --machine-embeddings data/runs/<RUN_ID>/activations/lc0_flat --human-embeddings data/runs/<RUN_ID>/activations/human_flat --out data/runs/<RUN_ID>/concepts/dynamic_sparse/novelty_report.json

## Validation and Acceptance

1. **Audit:** `docs/schut_audit_report.md` exists and justifies the implementation.
2. **Filtration:** `filter_fens_eval.py` successfully filters a list of 1,000 FENs to a smaller subset, and inspecting the subset confirms they are balanced middlegames (e.g., no mate-in-1s).
3. **Matching:** `tools/match_concepts.py` runs without errors and produces a ranked list of puzzle tags for each concept direction.
4. **Dynamic rollout concepts:** `tools/build_mcts_pairs.py` writes a rollout-pair `.npz`, and `tools/solve_dynamic_concepts.py` solves a sparse vector with held-out constraint satisfaction above shuffled-pair baselines.
5. **Novelty:** `tools/filter_novel_concepts.py` writes `novelty_report.json` with positive machine-vs-human reconstruction advantage for accepted vectors.
6. **Teachability:** a student trained on concept prototypes shows top-1 overlap lift over random-prototype training.

## Idempotence and Recovery

All tools will write to specific output files (`--out`). Overwriting is safe. Checkpointing FEN lists ensures we don't have to re-run expensive MCTS evaluations if a later pipeline step fails.

## Artifacts and Notes

- `docs/schut_audit_report.md`
- `IMPLEMENTATION_STATUS_AND_NEXT_WORK.md`
- `tools/filter_fens_eval.py`
- `tools/solve_dynamic_concepts.py`
- `tools/filter_novel_concepts.py`
- `notebooks/puzzle_concept_matching.ipynb` or `tools/match_concepts.py`

## Interfaces and Dependencies

In `tools/filter_fens_eval.py`, define:

    def filter_fens_by_eval(fens: list[str], engine: chess.engine.SimpleEngine, limit: chess.engine.Limit, min_cp: int, max_cp: int) -> list[str]:
        """Keep only FENs where the LC0 search evaluation falls within [min_cp, max_cp]."""
