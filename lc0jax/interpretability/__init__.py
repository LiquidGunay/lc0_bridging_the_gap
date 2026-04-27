"""Interpretability tooling for LC0 JAX."""

from .activations import dump_activations, project_token_activations, reshape_token_activations
from .concepts import (
    aggregate_trajectory,
    discover_concepts,
    dynamic_rollout_differences,
    patch_activations,
    solve_sparse_concept_from_differences,
)
from .datasets import (
    filter_activation_records_by_fens,
    filter_fens,
    filter_pgn,
    iter_activation_records,
    iter_fens,
    lichess_time_class,
    parse_time_control,
    pgn_to_activation_records,
    pgn_to_fens,
)
from .mcts_rollouts import (
    activation_records_for_line,
    build_rollout_pair_record,
    pv_to_fens,
)
from .novelty import novelty_curve, reconstruction_loss, right_svd_basis
from .pair_builders import (
    iter_rollout_pair_records,
    load_activation_index,
    materialize_rollout_differences,
)

__all__ = [
    "dump_activations",
    "project_token_activations",
    "reshape_token_activations",
    "aggregate_trajectory",
    "discover_concepts",
    "dynamic_rollout_differences",
    "patch_activations",
    "solve_sparse_concept_from_differences",
    "novelty_curve",
    "reconstruction_loss",
    "right_svd_basis",
    "filter_fens",
    "filter_activation_records_by_fens",
    "filter_pgn",
    "iter_activation_records",
    "iter_fens",
    "lichess_time_class",
    "parse_time_control",
    "pgn_to_activation_records",
    "pgn_to_fens",
    "build_rollout_pair_record",
    "activation_records_for_line",
    "pv_to_fens",
    "iter_rollout_pair_records",
    "load_activation_index",
    "materialize_rollout_differences",
]
