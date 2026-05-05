"""Causal policy-margin validation for dynamic rollout concepts."""

from __future__ import annotations

from typing import Any

import numpy as np


def policy_margin_report(
    *,
    base_policy: np.ndarray,
    patched_policy: np.ndarray,
    best_indices: list[int] | np.ndarray,
    subpar_indices: list[int] | np.ndarray,
    legal_masks: np.ndarray | None = None,
    root_fens: list[str],
    best_moves: list[str],
    subpar_moves: list[str],
) -> dict[str, Any]:
    """Summarize best-vs-subpar policy margin changes under patching."""
    base_policy = np.asarray(base_policy)
    patched_policy = np.asarray(patched_policy)
    best_indices = np.asarray(best_indices, dtype=np.int64)
    subpar_indices = np.asarray(subpar_indices, dtype=np.int64)
    if base_policy.shape != patched_policy.shape:
        raise ValueError(
            f"Policy shape mismatch: {base_policy.shape} vs {patched_policy.shape}"
        )
    if base_policy.ndim != 2:
        raise ValueError(f"Expected rank-2 policy logits, got {base_policy.shape}")
    if best_indices.shape != subpar_indices.shape:
        raise ValueError("best_indices and subpar_indices must have the same shape")
    if best_indices.shape[0] != base_policy.shape[0]:
        raise ValueError("Move index count must match policy batch size")
    if legal_masks is not None:
        legal_masks = np.asarray(legal_masks, dtype=bool)
        if legal_masks.shape != base_policy.shape:
            raise ValueError(
                f"Legal mask shape mismatch: {legal_masks.shape} vs {base_policy.shape}"
            )

    row_idx = np.arange(base_policy.shape[0])
    base_margin = base_policy[row_idx, best_indices] - base_policy[row_idx, subpar_indices]
    patched_margin = (
        patched_policy[row_idx, best_indices] - patched_policy[row_idx, subpar_indices]
    )
    delta_margin = patched_margin - base_margin
    if legal_masks is None:
        base_for_top = base_policy
        patched_for_top = patched_policy
    else:
        base_for_top = np.where(legal_masks, base_policy, -np.inf)
        patched_for_top = np.where(legal_masks, patched_policy, -np.inf)
    base_top = np.argmax(base_for_top, axis=1)
    patched_top = np.argmax(patched_for_top, axis=1)

    examples = []
    for idx in range(base_policy.shape[0]):
        examples.append(
            {
                "root_fen": root_fens[idx] if idx < len(root_fens) else "",
                "best_move": best_moves[idx] if idx < len(best_moves) else "",
                "subpar_move": subpar_moves[idx] if idx < len(subpar_moves) else "",
                "base_margin": float(base_margin[idx]),
                "patched_margin": float(patched_margin[idx]),
                "delta_margin": float(delta_margin[idx]),
                "base_top_index": int(base_top[idx]),
                "patched_top_index": int(patched_top[idx]),
                "top1_changed": bool(base_top[idx] != patched_top[idx]),
            }
        )

    return {
        "num_pairs": int(base_policy.shape[0]),
        "mean_base_margin": float(np.mean(base_margin)),
        "mean_patched_margin": float(np.mean(patched_margin)),
        "mean_delta_margin": float(np.mean(delta_margin)),
        "median_delta_margin": float(np.median(delta_margin)),
        "fraction_delta_positive": float(np.mean(delta_margin > 0)),
        "top1_change_rate": float(np.mean(base_top != patched_top)),
        "top1_legal_masked": bool(legal_masks is not None),
        "examples": examples,
    }


def control_calibration_report(
    target_report: dict[str, Any],
    control_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare a learned-direction policy-margin report to control directions."""
    if not control_reports:
        return {"control_count": 0}
    control_delta = np.asarray(
        [report["mean_delta_margin"] for report in control_reports],
        dtype=np.float64,
    )
    target_delta = float(target_report["mean_delta_margin"])
    return {
        "control_count": int(control_delta.shape[0]),
        "target_mean_delta_margin": target_delta,
        "mean_control_delta_margin": float(np.mean(control_delta)),
        "median_control_delta_margin": float(np.median(control_delta)),
        "max_control_delta_margin": float(np.max(control_delta)),
        "min_control_delta_margin": float(np.min(control_delta)),
        "target_minus_mean_control_delta_margin": float(target_delta - np.mean(control_delta)),
        "target_exceeds_control_fraction": float(np.mean(target_delta > control_delta)),
    }
