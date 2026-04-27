# Implementation Status and Next Work

Updated: 2026-04-27

## Current Implementation Status

The repo has a solid LC0 BT4 substrate: protobuf weight loading, JAX/Flax inference, ONNX oracle checks, LC0 input encoding, policy mapping, activation dumping, dataset filters, concept direction experiments, causal patching, and report generation.

The Schut-parity status is partial but now has an end-to-end dynamic smoke path. The existing concept discovery pipeline can solve static sparse separators over two activation datasets, and the new dynamic path can extract LC0 MCTS optimal/subpar rollout pairs, dump flat trajectory activations, materialize paired differences, split held-out roots, solve a sparse vector, and run novelty scoring. The remaining gap is scale and evaluation rigor, not basic plumbing.

Implemented toward parity:

- Activation dumps now expose `--activation-mode mean|flat`. `mean` preserves the prior pooled behavior; `flat` preserves square-local geometry by flattening the 64 board-token activations.
- Activation dumps can optionally store raw `[N, 64, channels]` token activations with `--store-token-activations` and policy logits with `--store-policy-logits`.
- `lc0jax.interpretability.concepts.solve_sparse_concept_from_differences` implements the L1 sparse soft-margin objective over paired differences.
- `lc0jax.interpretability.concepts.dynamic_rollout_differences` aggregates stored optimal/subpar rollout activations into `psi(tau+) - psi(tau-)` rows.
- `tools/solve_dynamic_concepts.py` solves active or reversed-sign sparse concepts from a stored rollout-pair `.npz` file.
- `lc0jax.interpretability.novelty` and `tools/filter_novel_concepts.py` implement the Schut-style SVD reconstruction comparison between machine and human activation bases.
- `lc0jax.interpretability.mcts_rollouts` and `tools/build_mcts_pairs.py` provide the first LC0 MultiPV rollout-pair builder. It writes JSONL records with root FEN, best PV, selected subpar PVs, centipawn scores, optional unique trajectory FENs, and preferred trajectory activation records with rolling history.
- `lc0jax.interpretability.pair_builders` and `tools/materialize_mcts_pairs.py` join rollout-pair JSONL records with trajectory activation shards and write solver-ready `pairs.npz` files containing `differences = psi(best) - psi(subpar)` plus aligned metadata. New trajectory records carry stable activation keys so repeated FENs under different histories do not collide.
- `lc0jax.interpretability.dynamic_reports` and `tools/build_dynamic_concept_report.py` build markdown report cards from `pairs.npz`, solver `report.json`, and optional `novelty_report.json`.
- `lc0jax.interpretability.dynamic_baselines` and `tools/dynamic_concept_baselines.py` compare learned dynamic directions against random sparse vectors, shuffled-label projections, and optional shuffled-label sparse solves.
- `lc0jax.interpretability.dynamic_evaluation` and `tools/evaluate_dynamic_concept.py` report held-out constraint and margin satisfaction for a learned dynamic direction on a held-out pair split. The CLI evaluates `raw_direction` by default so held-out margins are comparable to solver margins.
- `lc0jax.interpretability.dynamic_prototypes` and `tools/select_dynamic_prototypes.py` select top-scoring dynamic concept pair rows plus random controls for future teachability curricula. Prototype selection auto-detects reversed concept runs and preserves row-aligned metadata in selected rows.
- `lc0jax.interpretability.dynamic_causal` and `tools/dynamic_policy_margin.py` measure best-vs-subpar policy-logit margin changes before and after patching a dynamic concept direction.
- `lc0jax.interpretability.dynamic_splits` and `tools/split_dynamic_pairs.py` split dynamic `pairs.npz` files by root FEN without the fullmove counter, preserving known row-aligned arrays and scalar metadata while preventing same-position leakage across train/test.
- `tools/pgn_to_activation_records.py` writes JSONL records with rolling `history_fens`, and `tools/dump_activations.py --records` passes those boards to LC0 encoding instead of using empty history.
- `tools/run_full_pipeline.sh` now defaults to history-aware human activation records when the broadcast PGN is available; set `HISTORY_HUMAN_RECORDS=0` to keep the old FEN-only path.
- GCP smoke run `data/runs/gcp_dynamic_smoke_records_20260427` on `pipeline-vm` validated the full dynamic path from LC0 MultiPV search through history-aware flat activation dumping, `pairs.npz` materialization, sparse solve, and novelty reporting.

Known gaps:

- The rollout materializer writes direct difference matrices rather than padded `optimal_rollouts` and `subpar_rollouts` tensors. This is solver-ready and avoids ragged PV padding, but a future report builder may still want optional padded trajectory tensors for visualization.
- FEN-only activation dumps still call `encode_board(board, [])`; use `--records` for PGN-derived human games and MCTS trajectory dumps when history matters.
- Static puzzle-tag matching is useful for interpretation, but it is not the unsupervised discovery signal used by Schut et al.
- Teachability filtering is not implemented. We need a weaker LC0 checkpoint or student model, prototype curricula, KL distillation, and top-1 overlap lift against random-prototype baselines.
- Full-scale activation dumps, MCTS pair extraction, SVD sweeps on large matrices, and teachability training should run on GCP, not on this local workspace.

## Next Work Items

1. Scale the dynamic pipeline beyond the smoke run.
   Use a nontrivial root set, higher LC0 node budgets, sharding, and held-out pairs. Keep outputs under `data/runs/<RUN_ID>/` and record commands, LC0 version, model checksum, machine type, and node budget.

2. Run dynamic concept reports with causal policy-margin effects.
   The report tooling now includes roots, best/subpar moves, PVs, solver stats, pair materialization metadata, novelty summaries, baselines, and optional policy-margin patch summaries. Run this on larger dynamic datasets and include the report artifacts.

3. Use held-out dynamic concept splits in the next GCP run.
   The split tool now creates root-grouped train/test `pairs.npz` files, ignoring FEN fullmove counters for grouping. Solve on train, then run held-out evaluation, baselines, policy-margin patching, and reports on held-out test pairs.

4. Scale random and shuffled baselines.
   The baseline tool now supports random sparse vectors, shuffled-label projections, and optional shuffled sparse solves. Run it on larger dynamic datasets and add held-out train/test splits by root position.

5. Add teachability evaluation.
   Prototype selection now exists for dynamic concepts. Next, train a small student or weaker LC0 checkpoint on selected prototypes with KL to the teacher policy and report top-1 overlap lift against random prototype curricula.

## GCP Compute Policy

Use local compute only for unit tests, fixture-sized `.npz` files, and small smoke runs. Provision GCP for:

- LC0 MCTS pair extraction above notebook-scale node counts.
- Full BT4 activation dumps over large human or LC0 datasets.
- Large SVD novelty sweeps.
- Teachability training or distillation.

Every GCP run should write outputs under `data/runs/<RUN_ID>/` and record the machine type, accelerator, LC0 binary path/version, BT4 model checksum, node budget, and exact command line in the run directory.

## Validation Log

- 2026-04-27: `.venv/bin/python -m pytest -q` passed with 37 tests.
- 2026-04-27: CLI import smoke tests passed for `tools/build_mcts_pairs.py`, `tools/solve_dynamic_concepts.py`, and `tools/filter_novel_concepts.py`.
- 2026-04-27: CLI import smoke test passed for `tools/materialize_mcts_pairs.py`.
- 2026-04-27: CLI import smoke tests passed for `tools/pgn_to_activation_records.py` and the updated `tools/dump_activations.py --records` interface.
- 2026-04-27: CLI import smoke test passed for `tools/filter_activation_records.py`; `bash -n tools/run_full_pipeline.sh` passed.
- 2026-04-27: Initial GCP smoke run `gcp_dynamic_smoke_20260427` completed on `pipeline-vm` (`n2-standard-16` spot, CPU JAX, LC0 v0.32.1 built 2026-04-27), but PR review found that path used FEN-only trajectory activations and lost LC0 history context.
- 2026-04-27: The builder now writes history-aware `--out-trajectory-records`, activation shards store `activation_keys`, and materialization indexes by those keys when present.
- 2026-04-27: Replacement GCP smoke run `gcp_dynamic_smoke_records_20260427` completed with history-aware trajectory records. It kept 1 LC0 rollout pair from 20 root FENs, wrote 7 trajectory records, materialized `pairs.npz` with shape `(1, 65536)`, solved a flat sparse concept with CVXPY status `optimal`, and wrote `novelty_report.json` with `positive_rank_fraction=1.0` on the tiny smoke reference set.
- 2026-04-27: GCP policy-margin smoke ran `tools/dynamic_policy_margin.py` on `gcp_dynamic_smoke_records_20260427` with `alpha=0.1`, wrote `policy_margin_report.json`, and rebuilt `report.md` with the policy-margin section. The legal-masked toy one-pair sample had `mean_delta_margin=0.0` and `skipped_rows=0`, so this validates plumbing rather than concept strength.
- 2026-04-27: Held-out dynamic split tests passed; `.venv/bin/python -m pytest -q` passed with 58 tests after adding `tools/split_dynamic_pairs.py`. Review hardening added fullmove-insensitive grouping and explicit row-aligned key handling.
- 2026-04-27: Held-out dynamic evaluation tests passed; `.venv/bin/python -m pytest -q` passed with 62 tests after adding `tools/evaluate_dynamic_concept.py`. Review hardening made the CLI default to `raw_direction`, separated solver-pair and report-pair paths in markdown, and added explicit `--evaluation` report CLI coverage.
- 2026-04-27: Dynamic prototype selection tests passed; `.venv/bin/python -m pytest -q` passed with 66 tests after adding `tools/select_dynamic_prototypes.py`. Review hardening added reversed-concept scoring, original-space direction-key validation, custom row metadata propagation, and explicit `--prototypes` report CLI coverage.
- 2026-04-27: Ruff could not be run because it is not installed in the current `.venv`; line lengths were checked manually for the touched Python files.
