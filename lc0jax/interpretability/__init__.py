"""Interpretability tooling for LC0 JAX."""

from .activations import dump_activations, project_token_activations, reshape_token_activations
from .concepts import (
    aggregate_trajectory,
    discover_concepts,
    dynamic_rollout_differences,
    patch_activations,
    screen_sparse_concept_features,
    solve_screened_sparse_concept_from_differences,
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
from .dynamic_evaluation import dynamic_evaluation_report
from .dynamic_prototypes import (
    dynamic_prototype_report,
    projection_scores,
    select_random_indices,
    select_top_indices,
)
from .dynamic_reports import build_dynamic_concept_report
from .dynamic_splits import (
    infer_pair_row_count,
    root_fen_group_key,
    root_split_summary,
    split_pair_indices,
    subset_pairs_payload,
)
from .dynamic_teachability import teachability_curriculum_records
from .manifests import dynamic_roots_manifest, sha256_file
from .mcts_rollouts import (
    activation_records_for_line,
    board_from_root_history,
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
    "screen_sparse_concept_features",
    "solve_screened_sparse_concept_from_differences",
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
    "dynamic_evaluation_report",
    "dynamic_prototype_report",
    "projection_scores",
    "select_random_indices",
    "select_top_indices",
    "build_dynamic_concept_report",
    "teachability_curriculum_records",
    "infer_pair_row_count",
    "root_fen_group_key",
    "root_split_summary",
    "split_pair_indices",
    "subset_pairs_payload",
    "build_rollout_pair_record",
    "board_from_root_history",
    "activation_records_for_line",
    "pv_to_fens",
    "dynamic_roots_manifest",
    "sha256_file",
    "iter_rollout_pair_records",
    "load_activation_index",
    "materialize_rollout_differences",
]
