"""Activation capture utilities."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from lc0jax.modeling.encode import encode_board
from lc0jax.modeling.inference import forward

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None


def _pool_tokens(act: np.ndarray, *, batch: int) -> np.ndarray:
    act = act.reshape((batch, 64, -1))
    return act.mean(axis=1)


def dump_activations(
    params,
    dataset_iter,
    *,
    out_dir: str,
    layer: str = "trunk",
    batch_size: int = 64,
    shard_size: int = 2048,
    progress_every: int | None = None,
    total_fens: int | None = None,
):
    """Write activation embeddings and metadata shards to disk."""
    if chess is None:
        raise ImportError("python-chess is required for activation dumps.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    emb_acc: list[np.ndarray] = []
    wdl_acc: list[np.ndarray] = []
    mlh_acc: list[np.ndarray] = []
    fen_acc: list[str] = []
    processed = 0
    start_time = time.time()

    batch_planes: list[np.ndarray] = []
    batch_fens: list[str] = []

    def _format_eta(rate: float, remaining: int) -> str:
        if rate <= 0:
            return "eta=?"
        eta = remaining / rate
        return f"eta={eta:0.1f}s"

    def log_progress(final: bool = False) -> None:
        if progress_every is None and not final:
            return
        elapsed = max(time.time() - start_time, 1e-6)
        rate = processed / elapsed
        if total_fens:
            pct = 100.0 * processed / total_fens
            remaining = max(total_fens - processed, 0)
            eta = _format_eta(rate, remaining)
            msg = (
                f"Processed {processed}/{total_fens} ({pct:0.1f}%), "
                f"rate={rate:0.1f} pos/s, elapsed={elapsed:0.1f}s, {eta}"
            )
        else:
            msg = f"Processed {processed} positions, rate={rate:0.1f} pos/s, elapsed={elapsed:0.1f}s"
        print(msg, flush=True)

    def flush(shard_idx: int) -> None:
        if not emb_acc:
            return
        emb = np.concatenate(emb_acc, axis=0)
        wdl = np.concatenate(wdl_acc, axis=0)
        mlh = np.concatenate(mlh_acc, axis=0)
        fens = np.asarray(fen_acc, dtype=object)
        shard_file = out_path / f"shard_{shard_idx:04d}.npz"
        np.savez_compressed(shard_file, embeddings=emb, wdl=wdl, mlh=mlh, fens=fens, layer=layer)
        print(f"Wrote {shard_file} ({len(fens)} positions)", flush=True)

    for fen in dataset_iter:
        board = chess.Board(fen)
        batch_fens.append(fen)
        batch_planes.append(encode_board(board, [], planes_layout="nchw", input_format="INPUT_CLASSICAL_112_PLANE"))
        processed += 1
        if progress_every and processed % progress_every == 0:
            log_progress()
        if len(batch_planes) < batch_size:
            continue

        planes = np.stack(batch_planes, axis=0)
        _policy, wdl, mlh, activations = forward(params, planes, capture=True)
        if layer not in activations:
            raise KeyError(f"Layer '{layer}' not captured. Available: {sorted(activations)}")
        act = np.asarray(activations[layer])
        emb = _pool_tokens(act, batch=planes.shape[0])
        emb_acc.append(emb)
        wdl_acc.append(np.asarray(wdl))
        mlh_acc.append(np.asarray(mlh))
        fen_acc.extend(batch_fens)

        if len(fen_acc) >= shard_size:
            flush(shard_idx)
            shard_idx += 1
            emb_acc.clear()
            wdl_acc.clear()
            mlh_acc.clear()
            fen_acc.clear()

        batch_planes.clear()
        batch_fens.clear()

    if batch_planes:
        planes = np.stack(batch_planes, axis=0)
        _policy, wdl, mlh, activations = forward(params, planes, capture=True)
        if layer not in activations:
            raise KeyError(f"Layer '{layer}' not captured. Available: {sorted(activations)}")
        act = np.asarray(activations[layer])
        emb = _pool_tokens(act, batch=planes.shape[0])
        emb_acc.append(emb)
        wdl_acc.append(np.asarray(wdl))
        mlh_acc.append(np.asarray(mlh))
        fen_acc.extend(batch_fens)

    flush(shard_idx)
    log_progress(final=True)
    shard_count = len(list(out_path.glob("shard_*.npz")))
    done_file = out_path / "done.txt"
    done_file.write_text(
        f"positions={processed}\nshards={shard_count}\nlayer={layer}\n",
        encoding="utf-8",
    )
