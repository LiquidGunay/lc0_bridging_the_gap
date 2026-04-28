# AGENTS

Guidance for future work in this repository. Keep this file short, factual, and updated with important instructions, pitfalls, and external references that are useful to revisit.

## Working agreements

- Follow `ExecPLANS.md` and keep `PLANS.md` up to date with progress, decisions, surprises, and outcomes.
- Keep the repository structure minimal within the current folder layout. Add new code under the existing subpackages instead of creating new top-level folders.
- Keep `README.md` updated so the repo stays easy to understand and run. Update it whenever you add or move major features, tools, or workflows.
- Ensure reproducibility details are kept current in `README.md` (uv-based environment setup, GPU/CPU paths, and any pinned tool/model versions) before pushing updates.
- After running concept discovery, use `tools/causal_validate.py` to generate `causal_report.json` files and rebuild `data/concept_report.md` with `tools/build_concept_report.py`.
- Use the specific target network `BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz` as the default model for all tests and comparisons.
- Use the official LC0 binary and its bundled `leela2onnx` for ONNX export; record the exact LC0 release version and archive checksum.
- As of 2026-04-28, the latest official LC0 release is v0.32.1; update this pin if a newer release is published.
- The official v0.32.1 release has no Linux binary assets; build from source with Meson/Ninja when running on Linux.
- LC0 build artifact currently lives at `/tmp/lc0-src/build/release/lc0` and depends on its build directory for `libz.so`.
- Use PGN as the primary dataset input for human and LC0 self-play data, and support cached FEN lists as an intermediate format.
- Assume CUDA-enabled JAX for the primary path (CUDA 13 pip wheels by default), but keep CPU-only fallback working and tested.
- The diffusion-related work is out of scope and should not be added unless the user explicitly requests it.
- The full data pipeline now writes outputs under `data/runs/<RUN_ID>/` by default; use `SKIP_EXISTING=1` to resume without overwriting.

## Known pitfalls to avoid

- Do not guess LC0 plane layouts or policy mappings. Port the definitions from LC0 source code and keep the plane order and policy index list exactly aligned with the target network output shape.
- Always confirm the ONNX input layout (NCHW vs NHWC) and the policy output shape (1858 vs 8x8x73) before wiring masking or decoding.
- Treat WDL vs scalar value outputs carefully; do not assume a single scalar if the ONNX model outputs three logits.
- Preserve board orientation and side-to-move semantics exactly as LC0 expects. Do not flip planes unless LC0 does.
- Keep deterministic tests that catch castling rights, en passant encoding, and history-length edge cases.
- Do not assume GPU-only execution; always keep a CPU path working for tests and small runs.
- The BT4 target network uses `INPUT_CLASSICAL_112_PLANE` and outputs `POLICY_ATTENTION` with 1858 logits plus WDL (3) and MLH (1).
- The base system Python lacks `venv`/`pip` tooling (no `python3-venv`), so generating protobuf bindings with `protoc` or `grpcio-tools` may require a prebuilt venv or alternate workaround.
- Use `uv venv .venv` and `uv pip install --python .venv/bin/python ...` for dependency installs in this environment.
- Protobuf schemas have been vendored under `lc0jax/proto/`; prefer using those for weights parsing and keep them in sync with the pinned LC0 release.
- BT4’s attention policy head uses the `kAttnPolicyMap` mapping (stored locally as `lc0jax/policy_attn_map.txt`); do not use the 8x8x73 policy map for attention heads.
- LC0 stores dense weights in output-major order; use the ONNX exporter’s `{1,0}` transpose convention when mapping `.pb.gz` weights to JAX matrices.
- Multihead attention weights may omit `input_embedding`; LC0’s loader upgrades them to `INPUT_EMBEDDING_PE_DENSE` for attention-body nets.
- `tools/filter_pgn.py` filters PGNs by minimum rating and Lichess time-control class; Lichess defines time class using estimated duration = base + 40 * increment.
- Code is organized into `lc0jax/modeling`, `lc0jax/training`, `lc0jax/uci`, and `lc0jax/interpretability`; keep new code in the correct folder.
- Use `tools/download_data.py` to fetch the latest Lichess standard PGNs and LC0 training tarballs; the Lichess dumps are `.zst`, so decompression requires the `zstandard` Python package.
- Use `tools/download_data.py lichess-sample` to stream-filter the latest Lichess dump when a small sample (e.g., 1k games) is sufficient.
- Use `tools/filter_fens.py` when you want to drop opening positions; phase is normalized to `[0,1]` with `1.0` = opening and `0.0` = endgame.
- LC0 chunk-derived FENs use `fullmove_number=1`, so `min_ply` filters won’t work there; prefer phase/piece filters instead.
- Use `tools/download_data.py lichess-puzzles` for the puzzles dataset; by default it applies the first move to yield the position shown to the solver (`--raw-fen` keeps the original).
- `tools/filter_fens_disagreement.py` filters positions where LC0 search (UCI) disagrees with the raw policy head (JAX), mirroring the paper’s “policy disagreement” filter.
- Disagreement filtering is slow (LC0 search); use low nodes for quick tests and `--onnx models/BT4.onnx` to avoid JAX compile time.
- The disagreement filter supports sharding (`--shard-count/--shard-index`) and manual resume (`--start-line` + `--append`, optional `--state-file`).
- The Lichess monthly dump is sparse for 2400+ classical; a 200k rated-game sample yielded only 7 matches, so consider alternate sampling (larger scan or API-based top-player pulls).
- Prefer sampling 2400+ rapid + classical together unless a stricter classical-only slice is explicitly required.
- If 2400+ is too sparse, drop to 2000+ rapid/classical for the 1k-game target and use a smaller FEN subset (e.g., 5k positions) for quick activation dumps.
- `TESTS.md` documents the rationale behind the current test suite; keep it updated when adding new tests.
- Use `tools/export_pb.py` (template + LINEAR16) to export JAX BT4 params back into LC0 `.pb.gz` for UCI use.
- The BT4 template contains an empty `policy_head_map` entry; prune empty map entries before serializing.
- Concept clustering uses scikit-learn; keep the dependency in sync with `pyproject.toml`.
- As of 2026-04-28, the latest Lichess standard rated PGN dump is `lichess_db_standard_rated_2026-03.pgn.zst` (confirm via the index before hard-coding any month).
- Use `tools/lc0_benchmark.py` for quick LC0 throughput checks; record benchmark results if performance regressions appear.
- LC0 does not accept a `--uci` flag; launch the binary directly for UCI tools (`tools/elo_bench.py`, `tools/uci_ui.py`).
- For long searches, increase the python-chess UCI timeout (default 60s in the tools) to avoid `TimeoutError`.

## References for later lookup

- Schut paper (concept discovery): https://arxiv.org/abs/2310.16410
- Target LC0 network file: https://storage.lczero.org/files/networks-contrib/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz
- LC0 source repository (encoder, policy map, protobuf definitions, ONNX exporter): https://github.com/LeelaChessZero/lc0
- LC0 project site: https://lczero.org
- LC0 releases (official binaries + leela2onnx): https://github.com/LeelaChessZero/lc0/releases
- JAX installation guide: https://jax.readthedocs.io/en/latest/installation.html
- JAX CUDA wheel index: https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
- Lichess database (monthly rated PGNs): https://database.lichess.org/
- Lichess standard PGN index: https://database.lichess.org/standard/
- Lichess standard PGN checksums: https://database.lichess.org/standard/sha256sums.txt
- Lichess puzzle database: https://database.lichess.org/#puzzles
- Lichess broadcast API (PGN endpoints): https://lichess.org/api/broadcast
- Lichess API forum example for `api/games/user` export: https://lichess.org/forum/general-chess-discussion/exporting-pgns
- Lichess FAQ time controls: https://lichess.org/faq#time-controls
- Leela training data index (selfplay chunks): https://storage.lczero.org/files/training_data/
- LC0 benchmark instructions: https://lczero.org/dev/wiki/running-a-benchmark/
- LC0 benchmark flags/options: https://lczero.org/dev/wiki/lc0-options/
- LC0 training tar filenames sometimes use 4-digit times (HHMM), not 6-digit (HHMMSS).
