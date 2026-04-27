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
from .dynamic_baselines import (
    dynamic_baseline_report,
    evaluate_direction,
    random_sparse_directions,
)
from .dynamic_causal import policy_margin_report
from .dynamic_reports import build_dynamic_concept_report
from .dynamic_splits import (
    infer_pair_row_count,
    root_split_summary,
    split_pair_indices,
    subset_pairs_payload,
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
    "dynamic_baseline_report",
    "evaluate_direction",
    "random_sparse_directions",
    "policy_margin_report",
    "build_dynamic_concept_report",
    "infer_pair_row_count",
    "root_split_summary",
    "split_pair_indices",
    "subset_pairs_payload",
    "build_rollout_pair_record",
    "activation_records_for_line",
    "pv_to_fens",
    "iter_rollout_pair_records",
    "load_activation_index",
    "materialize_rollout_differences",
]
