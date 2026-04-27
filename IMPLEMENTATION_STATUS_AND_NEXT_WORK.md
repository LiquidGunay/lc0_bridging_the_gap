# Implementation Status and Next Work

Updated: 2026-04-27

## Current Implementation Status

The repo has a solid LC0 BT4 substrate: protobuf weight loading, JAX/Flax inference, ONNX oracle checks, LC0 input encoding, policy mapping, activation dumping, dataset filters, concept direction experiments, causal patching, and report generation.

The Schut-parity status is partial but now has an end-to-end dynamic path beyond the first smoke tests. The existing concept discovery pipeline can solve static sparse separators over two activation datasets, and the new dynamic path can extract LC0 MCTS optimal/subpar rollout pairs, dump flat trajectory activations, materialize paired differences, split held-out roots, solve sparse vectors, run held-out reports, select prototypes, export teachability curricula, and run policy-margin patch checks. The remaining gap is robust scale and evaluation rigor, not basic plumbing.

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
- `lc0jax.interpretability.dynamic_teachability` and `tools/export_teachability_curriculum.py` export prototype/control selections to JSONL records with target/contrast moves and provenance for downstream student-training jobs.
- `lc0jax.interpretability.dynamic_causal` and `tools/dynamic_policy_margin.py` measure best-vs-subpar policy-logit margin changes before and after patching a dynamic concept direction.
- `lc0jax.interpretability.dynamic_splits` and `tools/split_dynamic_pairs.py` split dynamic `pairs.npz` files by root FEN without the fullmove counter, preserving known row-aligned arrays and scalar metadata while preventing same-position leakage across train/test.
- `tools/pgn_to_activation_records.py` writes JSONL records with rolling `history_fens`, and `tools/dump_activations.py --records` passes those boards to LC0 encoding instead of using empty history.
- `tools/run_full_pipeline.sh` now defaults to history-aware human activation records when the broadcast PGN is available; set `HISTORY_HUMAN_RECORDS=0` to keep the old FEN-only path.
- GCP smoke run `data/runs/gcp_dynamic_smoke_records_20260427` on `pipeline-vm` validated the full dynamic path from LC0 MultiPV search through history-aware flat activation dumping, `pairs.npz` materialization, sparse solve, and novelty reporting.
- GCP larger run `data/runs/gcp_dynamic_large_20260427_174945` had 100 candidate evaluation roots available on `pipeline-vm` (`us-central1-a`, `n2-standard-16`, no accelerator, CPU/JAX path, LC0 Eigen backend), using LC0 at `/root/lc0-src/build/release/lc0` (`v0.32.1 built Apr 27 2026`), BT4 SHA256 `e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618`, 800 LC0 nodes, MultiPV 4, and `max_pairs=40`. It scanned 49 roots before hitting the cap, kept 40 rollout-pair records, wrote 948 history-aware trajectory records, and materialized 93 flat dynamic differences with a grouped split of 72 train / 21 held-out rows.
- The same larger run completed an end-to-end mean-pooled fallback report under `data/runs/gcp_dynamic_large_20260427_174945/concepts/dynamic_sparse_mean`: mean pair shape `(93, 1024)`, train `(72, 1024)`, test `(21, 1024)`, solver status `optimal`, train constraint satisfaction `1.0`, train margin satisfaction `0.75`, held-out constraint satisfaction `0.667`, held-out margin satisfaction `0.381`, 32 teachability curriculum rows, and policy-margin `mean_delta_margin=-2.42e-08` on 16 held-out rows at `alpha=0.1`.

Known gaps:

- The rollout materializer writes direct difference matrices rather than padded `optimal_rollouts` and `subpar_rollouts` tensors. This is solver-ready and avoids ragged PV padding, but a future report builder may still want optional padded trajectory tensors for visualization.
- FEN-only activation dumps still call `encode_board(board, [])`; use `--records` for PGN-derived human games and MCTS trajectory dumps when history matters.
- Static puzzle-tag matching is useful for interpretation, but it is not the unsupervised discovery signal used by Schut et al.
- Teachability filtering is not implemented. We need a weaker LC0 checkpoint or student model, prototype curricula, KL distillation, and top-1 overlap lift against random-prototype baselines.
- The direct flat sparse CVXPY/SCS solve does not yet scale comfortably. On the larger GCP run, solving 72 train rows over 65,536 flat features was still active after about 30 minutes on `n2-standard-16` and was terminated after the mean-pooled fallback completed. Keep the flat pair artifacts, but add a faster solver path before larger flat sweeps.
- The first mean-pooled policy-margin patch effect was effectively zero at `alpha=0.1`. This validates the patch/report plumbing on held-out rows, but not concept strength or causal usefulness.
- Full-scale activation dumps, MCTS pair extraction, SVD sweeps on large matrices, and teachability training should run on GCP, not on this local workspace.

## Next Work Items

1. Add a faster large-flat dynamic solver path.
   The next implementation step should make 65,536-dimensional flat dynamic solves practical. Candidate approaches: feature screening by univariate margins before CVXPY, an sklearn sparse linear SVM/logistic fallback, SCS iteration/time-limit controls, token/channel grouped reductions, or a two-stage mean-to-flat refinement. Keep the exact Schut L1 objective available for small runs.

2. Re-run the larger flat report after the solver bottleneck is addressed.
   Reuse or regenerate a run like `gcp_dynamic_large_20260427_174945`, solve on the root-grouped train split, then run held-out evaluation, baselines, prototype selection, curriculum export, and policy-margin patching on the held-out test split.

3. Improve causal patch calibration.
   Sweep `alpha`, compare normalized `direction` vs `raw_direction`, and report policy-margin/top-1 changes against random and shuffled controls. The first larger mean-pooled run had near-zero margin movement, so this needs quantitative calibration before teachability claims.

4. Scale random and shuffled baselines.
   The baseline tool now supports random sparse vectors, shuffled-label projections, and optional shuffled sparse solves. Run it on larger dynamic datasets and add held-out train/test splits by root position.

5. Add teachability evaluation.
   Prototype and curriculum export now exist for dynamic concepts. Next, train a small student or weaker LC0 checkpoint on selected prototypes with KL to the teacher policy and report top-1 overlap lift against random prototype curricula.

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
- 2026-04-27: Teachability curriculum export tests passed; `.venv/bin/python -m pytest -q` passed with 72 tests after adding `tools/export_teachability_curriculum.py`. Review hardening added required row-field validation, provenance in each curriculum row, and negative-limit CLI coverage.
- 2026-04-27: Ruff could not be run because it is not installed in the current `.venv`; line lengths were checked manually for the touched Python files.
- 2026-04-27: Larger GCP dynamic validation `gcp_dynamic_large_20260427_174945` ran on `pipeline-vm` with LC0 nodes `800`, MultiPV `4`, and `max_pairs=40`. MCTS kept 40 pair records from 49 scanned roots and produced 948 history-aware trajectory records. Flat materialization produced `(93, 65536)` differences split into 72 train / 21 test rows, but the flat CVXPY/SCS solve was still running after about 30 minutes and was terminated. A mean-pooled fallback completed end-to-end with `(93, 1024)` differences, solver status `optimal`, held-out constraint satisfaction `0.667`, held-out margin satisfaction `0.381`, 32 curriculum rows, and policy-margin `mean_delta_margin=-2.42e-08` at `alpha=0.1`. Exact command lines and environment metadata are recorded in `data/runs/gcp_dynamic_large_20260427_174945/RUN_METADATA.md` on `pipeline-vm`; refreshed local artifact bundle: `/tmp/gcp_dynamic_large_20260427_174945_mean_artifacts.tar.gz`.
