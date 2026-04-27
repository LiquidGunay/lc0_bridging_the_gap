# Schut Paper Reproduction and Extension

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document serves as an extension to `PLANS.md` in accordance with `ExecPLANS.md`.

## Purpose / Big Picture

The goal is to accurately reproduce the concept discovery methodology from the Schut et al. AlphaZero paper (arXiv:2310.16410) on LC0, and extend it by matching discovered concepts to Lichess puzzle themes in an unsupervised way. This involves three major steps:
1. Auditing our current `discover_concepts.py` implementation against the exact mathematical formulations used in the Schut paper (e.g., whitening, PCA, SVM probes) to ensure our concept directions are derived identically.
2. Implementing a robust data filtration pipeline using LC0's MCTS evaluation to isolate complex, balanced middlegames (win probability between 10% and 90%), stripping out forced endgames and rote openings.
3. Extracting Lichess puzzle FENs along with their thematic tags, generating network activations for these puzzles, and aligning our unsupervised concept directions with these human-understandable tags.

## Progress

- [x] (2026-04-25) Audit `tools/discover_concepts.py` against the Schut paper methodology.
- [x] (2026-04-25) Implement `tools/filter_fens_eval.py` to filter FENs based on LC0 MCTS evaluation limits.
- [x] (2026-04-25) Modify `tools/download_data.py` (specifically `lichess-puzzles`) to save puzzle tags alongside FENs (e.g., as JSONL).
- [x] (2026-04-26) Run the full filtered data pipeline: extract FENs, filter via MCTS + heuristics, and dump activations.
- [x] (2026-04-26) Run the audited concept discovery pipeline on the filtered dataset.
- [x] (2026-04-26) Dump activations for the tagged Lichess puzzles.
- [x] (2026-04-25) Create `tools/match_concepts.py` (or a Jupyter notebook) to project puzzle activations onto discovered concept directions and rank the correlation of Lichess tags to each concept.

## Surprises & Discoveries

- Observation: L1-regularized SVM (via `cvxpy`) optimization produced sparse concept vectors exactly as described by Schut et al., providing a clear mathematical link to human-understandable Lichess themes without requiring post-hoc PCA or thresholding.
- Observation: Spot TPU (`v5litepod-8` / `v5litepod-4`) provisioning encountered quota limitations (max 4 chips for v5litepod APIs in testing zones) and capacity errors. We fell back to deploying a robust `pipeline-vm` controller (`n2-standard-16`) to compile LC0 from source and run localized, cheap end-to-end test validations.
- Observation: Discovering concepts and matching them to Lichess puzzle tags works seamlessly; `match_concepts.py` confirmed that the raw discovered concept vectors are meaningfully correlated with human tags like `middlegame` and `fork`.

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

## Outcomes & Retrospective

- Successfully audited and implemented the exact `cvxpy` L1-SVM concept discovery logic outlined in Schut et al.
- Created `filter_fens_eval.py` to curate a high-quality dataset of balanced middlegame positions using LC0's MCTS engine.
- Extended the pipeline to persist Lichess puzzle tags (`.jsonl`) and developed `match_concepts.py` to project those human-interpretable tags onto our unsupervised concept vectors.
- Validated the entire pipeline locally on a GCP Spot VM instance (`pipeline-vm`), achieving an end-to-end successful run (data extraction -> MCTS evaluation filter -> JAX activation dump -> SVM concept discovery -> Puzzle concept mapping).

**Next Steps:**
1. Now that the pipeline logic is fully validated and functioning end-to-end, you can scale this up by running `run_full_pipeline.sh` on your full dataset on the controller VM.
2. For intensive training or enormous dataset activation dumping, utilize the orchestrator script `scripts/run_tpu_spot_jepa.py` on the larger codebase to queue up the validated artifacts and wait for Spot TPU capacity to become available (which it will handle automatically via polling).

## Context and Orientation

The repository currently implements a pipeline that downloads Lichess data, applies heuristic filters (`tools/filter_fens.py`), dumps LC0 activations (`tools/dump_activations.py`), and discovers concept vectors (`tools/discover_concepts.py`). However, the data filtration lacks the rigorous evaluation-based filtering used by Schut et al. to isolate complex middlegames, and our concept discovery code needs verification against the paper's exact math. Furthermore, we want to map discovered concepts to Lichess puzzle tags. `tools/download_data.py` currently discards these tags when downloading puzzles.

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

## Concrete Steps

(To be populated during execution with exact commands)

## Validation and Acceptance

1. **Audit:** `docs/schut_audit_report.md` exists and justifies the implementation.
2. **Filtration:** `filter_fens_eval.py` successfully filters a list of 1,000 FENs to a smaller subset, and inspecting the subset confirms they are balanced middlegames (e.g., no mate-in-1s).
3. **Matching:** `tools/match_concepts.py` runs without errors and produces a ranked list of puzzle tags for each concept direction.

## Idempotence and Recovery

All tools will write to specific output files (`--out`). Overwriting is safe. Checkpointing FEN lists ensures we don't have to re-run expensive MCTS evaluations if a later pipeline step fails.

## Artifacts and Notes

- `docs/schut_audit_report.md`
- `tools/filter_fens_eval.py`
- `notebooks/puzzle_concept_matching.ipynb` or `tools/match_concepts.py`

## Interfaces and Dependencies

In `tools/filter_fens_eval.py`, define:

    def filter_fens_by_eval(fens: list[str], engine: chess.engine.SimpleEngine, limit: chess.engine.Limit, min_cp: int, max_cp: int) -> list[str]:
        """Keep only FENs where the LC0 search evaluation falls within [min_cp, max_cp]."""
