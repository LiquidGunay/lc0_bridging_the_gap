"""Filter FENs where LC0 search disagrees with raw network policy."""

from __future__ import annotations

import argparse
from typing import Callable, Iterable

import numpy as np

import chess
import chess.engine
import onnxruntime as ort

from lc0jax.modeling.encode import encode_board
from lc0jax.modeling.inference import forward
from lc0jax.modeling.policy import attention_policy_map, legal_move_mask, policy_index_to_move
from lc0jax.modeling.weights import load_pb_gz, map_bt4_weights


def iter_fens(path: str, *, start_line: int = 0, shard_index: int = 0, shard_count: int = 1) -> Iterable[str]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    with open(path, "r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx < start_line:
                continue
            if shard_count > 1 and (line_idx % shard_count) != shard_index:
                continue
            fen = line.strip()
            if fen:
                yield fen


def _prepare_boards(fens: list[str]) -> tuple[list[str], list[chess.Board]]:
    boards: list[chess.Board] = []
    valid_fens: list[str] = []

    for fen in fens:
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        boards.append(board)
        valid_fens.append(fen)
    return valid_fens, boards

def _select_policy_moves(
    fens: list[str],
    boards: list[chess.Board],
    policy_logits: np.ndarray,
) -> list[tuple[str, chess.Board, chess.Move]]:
    results = []
    for fen, board, logits in zip(fens, boards, policy_logits):
        mask = legal_move_mask(board, "lc0_1858").astype(bool)
        masked = np.where(mask, logits, -1e9)
        idx = int(masked.argmax())
        move = policy_index_to_move(idx, "lc0_1858")
        results.append((fen, board, move))
    return results


def batch_policy_moves_jax(fens: list[str], params) -> list[tuple[str, chess.Board, chess.Move]]:
    valid_fens, boards = _prepare_boards(fens)
    if not boards:
        return []
    planes = [encode_board(board, [], planes_layout="nchw", input_format="INPUT_CLASSICAL_112_PLANE") for board in boards]
    planes_np = np.stack(planes).astype(np.float32)
    policy, _, _ = forward(params, planes_np)
    policy_np = np.asarray(policy)
    return _select_policy_moves(valid_fens, boards, policy_np)


def make_onnx_policy_fn(onnx_path: str) -> Callable[[list[str]], list[tuple[str, chess.Board, chess.Move]]]:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_info = sess.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    def policy_fn(fens: list[str]) -> list[tuple[str, chess.Board, chess.Move]]:
        valid_fens, boards = _prepare_boards(fens)
        if not boards:
            return []
        planes = [encode_board(board, [], planes_layout="nchw", input_format="INPUT_CLASSICAL_112_PLANE") for board in boards]
        data = np.stack(planes).astype(np.float32)
        if len(input_shape) == 4:
            c_dim = input_shape.index("C") if "C" in input_shape else None
            if c_dim is None:
                if input_shape[-1] == data.shape[1]:
                    data = np.transpose(data, (0, 2, 3, 1))
            elif c_dim == 3:
                data = np.transpose(data, (0, 2, 3, 1))
        if input_info.type == "tensor(float16)":
            data = data.astype(np.float16, copy=False)
        outputs = sess.run(None, {input_name: data})
        output_names = [out.name for out in sess.get_outputs()]
        out_map = {name: value for name, value in zip(output_names, outputs)}
        if "/output/policy" in out_map:
            policy_logits = out_map["/output/policy"]
        elif "output/policy" in out_map:
            policy_logits = out_map["output/policy"]
        else:
            policy_logits = out_map.get("policy")
        if policy_logits is None:
            raise RuntimeError(f"Policy output not found in ONNX outputs: {output_names}")
        return _select_policy_moves(valid_fens, boards, policy_logits)

    return policy_fn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pb", default=None, help="LC0 .pb.gz weights for raw policy (no search).")
    parser.add_argument("--onnx", default=None, help="Optional ONNX model for raw policy (faster, avoids JAX).")
    parser.add_argument("--lc0", required=True, help="Path to LC0 binary.")
    parser.add_argument("--weights", default=None, help="Weights for LC0 search (defaults to --pb).")
    parser.add_argument("--nodes", type=int, default=800)
    parser.add_argument("--movetime-ms", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--backend-opts", default=None)
    parser.add_argument("--uci-timeout", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--start-line", type=int, default=0, help="Skip the first N lines of the input FEN file.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for parallel runs.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total shard count for parallel runs.")
    parser.add_argument("--append", action="store_true", help="Append to output file instead of overwriting.")
    parser.add_argument(
        "--state-file",
        default=None,
        help="Optional state file to record progress (JSON with seen/kept).",
    )
    parser.add_argument("--state-every", type=int, default=None, help="Write state every N seen positions.")
    args = parser.parse_args()

    if args.nodes is None and args.movetime_ms is None:
        raise ValueError("Provide --nodes or --movetime-ms (or both).")

    weights_path = args.weights or args.pb

    policy_fn: Callable[[list[str]], list[tuple[str, chess.Board, chess.Move]]]
    params = None
    if args.onnx:
        policy_fn = make_onnx_policy_fn(args.onnx)
    else:
        if not args.pb:
            raise ValueError("--pb is required when --onnx is not provided.")
        bundle = load_pb_gz(args.pb)
        params = map_bt4_weights(bundle, mapping_table=attention_policy_map())
        policy_fn = lambda fens: batch_policy_moves_jax(fens, params)

    engine = chess.engine.SimpleEngine.popen_uci([args.lc0], timeout=args.uci_timeout)
    options = {}
    if weights_path:
        options["WeightsFile"] = weights_path
    if args.threads:
        options["Threads"] = args.threads
    if args.backend:
        options["Backend"] = args.backend
    if args.backend_opts:
        options["BackendOptions"] = args.backend_opts
    if options:
        engine.configure(options)

    limit = chess.engine.Limit(
        nodes=args.nodes,
        time=None if args.movetime_ms is None else args.movetime_ms / 1000.0,
    )

    seen = 0
    kept = 0
    batch: list[str] = []

    out_mode = "a" if args.append else "w"
    with open(args.out, out_mode, encoding="utf-8") as out_f:
        try:
            for fen in iter_fens(
                args.fens,
                start_line=args.start_line,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            ):
                batch.append(fen)
                if len(batch) < args.batch_size:
                    continue
                seen, kept, done = _process_batch(batch, policy_fn, engine, limit, out_f, seen, kept, args)
                if done:
                    batch = []
                    break
                batch = []
            if batch:
                seen, kept, _ = _process_batch(batch, policy_fn, engine, limit, out_f, seen, kept, args)
        finally:
            engine.quit()

    if args.state_file:
        _write_state(args.state_file, seen=seen, kept=kept)

    print(f"Disagreements kept: {kept} (seen {seen})")
    return 0


def _process_batch(
    batch: list[str],
    policy_fn: Callable[[list[str]], list[tuple[str, chess.Board, chess.Move]]],
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    out_f,
    seen: int,
    kept: int,
    args,
) -> tuple[int, int, bool]:
    for fen, board, policy_move in policy_fn(batch):
        seen += 1
        try:
            result = engine.play(board, limit)
        except Exception:
            continue
        search_move = result.move
        if search_move is None:
            continue
        if search_move.uci() != policy_move.uci():
            out_f.write(fen + "\n")
            kept += 1
        if args.progress_every and seen % args.progress_every == 0:
            print(f"Seen {seen} positions, kept {kept}", flush=True)
        if args.state_file:
            state_every = args.state_every or args.progress_every
            if state_every and seen % state_every == 0:
                _write_state(args.state_file, seen=seen, kept=kept)
        if args.max_positions is not None and kept >= args.max_positions:
            return seen, kept, True
    return seen, kept, False


def _write_state(path: str, *, seen: int, kept: int) -> None:
    import json
    from datetime import datetime, timezone

    payload = {
        "seen": seen,
        "kept": kept,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    raise SystemExit(main())
