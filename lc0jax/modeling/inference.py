"""Inference wrappers for LC0 Flax models."""

from __future__ import annotations

import jax.numpy as jnp

from .model import Bt4Model

def forward(params, planes, *, mask=None, capture: bool = False, patch: dict | None = None):
    """Run Flax forward pass and apply legality masking if provided."""
    model = Bt4Model()
    outputs = model.apply({}, planes, params, capture=capture, patch=patch)

    if capture:
        policy, wdl, moves_left, activations = outputs
    else:
        policy, wdl, moves_left = outputs
        activations = None

    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=bool)
        if mask_arr.ndim == 1:
            mask_arr = mask_arr[None, :]
        policy = jnp.where(mask_arr, policy, -1e9)

    if capture:
        return policy, wdl, moves_left, activations
    return policy, wdl, moves_left
