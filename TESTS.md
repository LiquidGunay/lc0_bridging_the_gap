# Test Rationale

This document explains why each test exists and what failure it is meant to catch.

## Encoder + policy tests

- `tests/test_encoder.py` validates that the LC0 plane encoder is deterministic, handles castling and en-passant correctly, and respects side-to-move semantics. These are the most common sources of silent model mismatches, so the tests are designed to fail if planes are mis-ordered or flipped.
- `tests/test_policy_legality.py` ensures the policy mapping and legality mask never choose illegal moves. This guards against mismatched policy index orders or missing move entries.

## Oracle equivalence

- `tests/test_oracle_equivalence.py` checks that ONNX oracle outputs have the expected shapes and are finite for a set of FENs. This catches regressions in the encoder, input layout, or ONNX runner wiring without requiring bit-for-bit parity.

## Dataset parsing

- `tests/test_datasets.py` verifies the Lichess index parsing helpers and time-class classification (based on base + 40 * increment).
- The stream-filter tests ensure multi-class filtering works (e.g., rapid + classical) and that rating/time-control filters are applied consistently. These tests prevent silent mis-filtering of human datasets.
- The activation-record tests verify that PGN conversion preserves rolling `history_fens` metadata for LC0 history-aware encoding, and that record JSONL can be filtered back to a selected FEN list after evaluation or disagreement filters.
- The FEN filter tests cover min-ply and phase-based filtering so we can exclude opening positions without breaking downstream activation dumps.
- Broadcast downloads are exercised via `tools/download_data.py lichess-broadcasts` (no unit test yet because it depends on network access).
- Puzzle downloads and the LC0 disagreement filter are not unit-tested; validate via the CLI tools because they require large files and an LC0 binary.

## Concept discovery

- `tests/test_concepts.py` verifies shape and score outputs for covariance-shift and clustering concept methods so we catch regressions in multi-vector outputs early.
- The sparse CVXPY paired-difference test checks the Schut-style objective directly: a positive-vs-negative difference matrix should produce a sparse direction with positive held-in constraints.
- The dynamic rollout-difference test checks the first non-search piece of dynamic concept discovery: stored optimal/subpar trajectories can be aggregated with both-player or single-player indexing before solving the sparse objective.
- `tests/test_mcts_rollouts.py` checks PV replay, side-to-move centipawn conversion, and serialization of python-chess analysis info without launching LC0.
- `tests/test_pair_builders.py` checks that rollout-pair JSONL plus activation shards materialize into the `differences` matrix consumed by the dynamic sparse solver.
- `tests/test_build_mcts_pairs.py` checks that MCTS pair extraction fails fast by default on expected per-position errors and only skips them when `--skip-errors` is explicit.
- `tests/test_dynamic_reports.py` checks that dynamic report cards include solver status, pair materialization metadata, novelty summaries, and best/subpar PV examples.
- `tests/test_dynamic_baselines.py` checks constraint-satisfaction metrics plus random sparse and shuffled-label baseline summaries.
- `tests/test_dynamic_evaluation.py` checks held-out dynamic direction evaluation reports and that the CLI defaults to `raw_direction` for margin metrics.
- `tests/test_dynamic_causal.py` checks best-vs-subpar policy-margin summaries from base and patched logits.
- `tests/test_dynamic_prototypes.py` checks score-based prototype selection, reversed concept scoring, random controls, metadata propagation, direction-key validation, and the prototype-selection CLI.
- `tests/test_dynamic_teachability.py` checks JSONL curriculum export from prototype/control reports, including provenance and malformed-row failures.
- `tests/test_dynamic_splits.py` checks root-grouped train/test splitting for dynamic `pairs.npz` files, including fullmove-insensitive grouping, metadata preservation, explicit custom row keys, and CLI output.
- `tests/test_model_patch.py` checks that channel-only and flat square-local concept directions can be reshaped for token-shaped model patch points.
- `tests/test_activations.py` checks that captured BT4 token activations can be reshaped to `[batch, 64, channels]` and projected either by mean pooling or by flattening square-local tokens.
- `tests/test_novelty.py` checks the SVD reconstruction novelty metric used to compare machine-game and human-game activation bases.
- The causal validation tool (`tools/causal_validate.py`) is intentionally not exercised in unit tests because it requires full model weights and GPU/CPU runtimes; validate it via the generated `data/concept_report.md` outputs instead.

## Training chunk parsing

- `tests/test_chunks.py` validates that a minimal chunk record converts into a sane board, and that policy indices map back to moves. This catches errors in bitboard transforms, castling handling, and policy mapping.
- `tests/test_training_downloads.py` ensures training tar filenames with 4- or 6-digit times are parsed, and that the latest entry is picked correctly.

## Export path

- `tests/test_export.py` checks LINEAR16 export roundtrips for vectors and matrices, ensuring the LC0 quantization and reshape conventions are preserved before writing `.pb.gz` weights.

## Elo bench

- `tests/test_elo_bench.py` validates the Elo-from-score conversion used in `tools/elo_bench.py` so the reported Elo diff stays consistent when refactoring the benchmark tool.

## Coverage gaps and intent

- These tests are not full functional parity tests against LC0; they are designed to catch the most likely mismatches early.
- If we change encoder formats, policy mapping, or chunk parsing, add or adjust tests here first to keep regressions visible.
