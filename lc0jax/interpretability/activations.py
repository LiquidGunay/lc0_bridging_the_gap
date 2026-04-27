"""Activation capture utilities."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Literal

import numpy as np

from lc0jax.modeling.encode import encode_board
from lc0jax.modeling.inference import forward

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None

ActivationMode = Literal["mean", "flat"]


def reshape_token_activations(act: np.ndarray, *, batch: int) -> np.ndarray:
    """Return captured board-token activations as ``[batch, 64, channels]``."""
    act = np.asarray(act)
    if act.ndim == 3:
        if act.shape[0] != batch or act.shape[1] != 64:
            raise ValueError(
                "Expected activation shape [batch, 64, channels], "
                f"got {act.shape} for batch={batch}"
            )
        return act
    if act.ndim != 2:
        raise ValueError(f"Expected rank-2 or rank-3 activations, got shape {act.shape}")
    if act.shape[0] != batch * 64:
        raise ValueError(
            f"Expected first dimension batch*64={batch * 64}, got {act.shape[0]}"
        )
    return act.reshape((batch, 64, -1))


def project_token_activations(tokens: np.ndarray, *, mode: ActivationMode) -> np.ndarray:
    """Project ``[batch, 64, channels]`` tokens into a concept-ready embedding matrix."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 3 or tokens.shape[1] != 64:
        raise ValueError(
            "Expected token activations shaped [batch, 64, channels], "
            f"got {tokens.shape}"
        )
    if mode == "mean":
        return tokens.mean(axis=1)
    if mode == "flat":
        return tokens.reshape((tokens.shape[0], -1))
    raise ValueError(f"Unsupported activation mode: {mode}")


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
    activation_mode: ActivationMode = "mean",
    store_token_activations: bool = False,
    store_policy_logits: bool = False,
):
    """Write activation embeddings and metadata shards to disk."""
    if chess is None:
        raise ImportError("python-chess is required for activation dumps.")
    if activation_mode not in {"mean", "flat"}:
        raise ValueError(f"Unsupported activation mode: {activation_mode}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    emb_acc: list[np.ndarray] = []
    token_acc: list[np.ndarray] = []
    policy_acc: list[np.ndarray] = []
    wdl_acc: list[np.ndarray] = []
    mlh_acc: list[np.ndarray] = []
    fen_acc: list[str] = []
    history_acc: list[list[str]] = []
    game_id_acc: list[str] = []
    ply_acc: list[int] = []
    activation_key_acc: list[str] = []
    processed = 0
    start_time = time.time()

    batch_planes: list[np.ndarray] = []
    batch_fens: list[str] = []
    batch_history_fens: list[list[str]] = []
    batch_game_ids: list[str] = []
    batch_plies: list[int] = []
    batch_activation_keys: list[str] = []

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
            msg = (
                f"Processed {processed} positions, rate={rate:0.1f} pos/s, "
                f"elapsed={elapsed:0.1f}s"
            )
        print(msg, flush=True)

    def flush(shard_idx: int) -> None:
        if not emb_acc:
            return
        emb = np.concatenate(emb_acc, axis=0)
        wdl = np.concatenate(wdl_acc, axis=0)
        mlh = np.concatenate(mlh_acc, axis=0)
        fens = np.asarray(fen_acc, dtype=object)
        shard_file = out_path / f"shard_{shard_idx:04d}.npz"
        payload = {
            "schema_version": np.asarray(2, dtype=np.int32),
            "embeddings": emb,
            "wdl": wdl,
            "mlh": mlh,
            "fens": fens,
            "history_fens": np.asarray(history_acc, dtype=object),
            "game_ids": np.asarray(game_id_acc, dtype=object),
            "plies": np.asarray(ply_acc, dtype=np.int32),
            "activation_keys": np.asarray(activation_key_acc, dtype=object),
            "layer": np.asarray(layer),
            "activation_mode": np.asarray(activation_mode),
        }
        if token_acc:
            payload["token_activations"] = np.concatenate(token_acc, axis=0)
        if policy_acc:
            payload["policy_logits"] = np.concatenate(policy_acc, axis=0)
        np.savez_compressed(shard_file, **payload)
        print(f"Wrote {shard_file} ({len(fens)} positions)", flush=True)

    def append_batch() -> None:
        planes = np.stack(batch_planes, axis=0)
        policy, wdl, mlh, activations = forward(params, planes, capture=True)
        if layer not in activations:
            raise KeyError(f"Layer '{layer}' not captured. Available: {sorted(activations)}")
        tokens = reshape_token_activations(np.asarray(activations[layer]), batch=planes.shape[0])
        emb = project_token_activations(tokens, mode=activation_mode)
        emb_acc.append(emb)
        if store_token_activations:
            token_acc.append(tokens)
        if store_policy_logits:
            policy_acc.append(np.asarray(policy))
        wdl_acc.append(np.asarray(wdl))
        mlh_acc.append(np.asarray(mlh))
        fen_acc.extend(batch_fens)
        history_acc.extend(batch_history_fens)
        game_id_acc.extend(batch_game_ids)
        ply_acc.extend(batch_plies)
        activation_key_acc.extend(batch_activation_keys)

    def normalize_item(item) -> tuple[str, list[str], str, int, str]:
        if isinstance(item, str):
            return item, [], "", -1, ""
        if not isinstance(item, dict):
            raise TypeError(f"Unsupported activation dataset item: {type(item)!r}")
        fen = item["fen"]
        history_fens = [str(history_fen) for history_fen in item.get("history_fens", [])]
        game_id = str(item.get("game_id", ""))
        ply = int(item.get("ply", -1))
        activation_key = str(item.get("activation_key", ""))
        return fen, history_fens, game_id, ply, activation_key

    for item in dataset_iter:
        fen, history_fens, game_id, ply, activation_key = normalize_item(item)
        board = chess.Board(fen)
        batch_fens.append(fen)
        batch_history_fens.append(history_fens)
        batch_game_ids.append(game_id)
        batch_plies.append(ply)
        batch_activation_keys.append(activation_key)
        history_boards = [chess.Board(history_fen) for history_fen in history_fens]
        batch_planes.append(
            encode_board(
                board,
                history_boards,
                planes_layout="nchw",
                input_format="INPUT_CLASSICAL_112_PLANE",
            )
        )
        processed += 1
        if progress_every and processed % progress_every == 0:
            log_progress()
        if len(batch_planes) < batch_size:
            continue

        append_batch()

        if len(fen_acc) >= shard_size:
            flush(shard_idx)
            shard_idx += 1
            emb_acc.clear()
            token_acc.clear()
            policy_acc.clear()
            wdl_acc.clear()
            mlh_acc.clear()
            fen_acc.clear()
            history_acc.clear()
            game_id_acc.clear()
            ply_acc.clear()
            activation_key_acc.clear()

        batch_planes.clear()
        batch_fens.clear()
        batch_history_fens.clear()
        batch_game_ids.clear()
        batch_plies.clear()
        batch_activation_keys.clear()

    if batch_planes:
        append_batch()

    flush(shard_idx)
    log_progress(final=True)
    shard_count = len(list(out_path.glob("shard_*.npz")))
    done_file = out_path / "done.txt"
    done_file.write_text(
        "\n".join(
            [
                f"positions={processed}",
                f"shards={shard_count}",
                f"layer={layer}",
                f"activation_mode={activation_mode}",
                f"store_token_activations={int(store_token_activations)}",
                f"store_policy_logits={int(store_policy_logits)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
