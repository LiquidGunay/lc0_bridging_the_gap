import json

import numpy as np
import pytest

from lc0jax.interpretability.dynamic_reports import build_dynamic_concept_report
from tools import build_dynamic_concept_report as build_dynamic_concept_report_cli


def test_build_dynamic_concept_report_includes_solver_novelty_and_pairs(tmp_path):
    pairs = tmp_path / "pairs.npz"
    metadata = {
        "activation_key": "token_activations",
        "num_activation_items": 4,
        "records_consumed": 1,
        "records_or_lines_skipped": 0,
    }
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 8), dtype=np.float32),
        root_fens=np.asarray(["root fen"], dtype=object),
        best_moves=np.asarray(["e2e4"], dtype=object),
        subpar_moves=np.asarray(["d2d4"], dtype=object),
        best_score_cp=np.asarray([50], dtype=object),
        subpar_score_cp=np.asarray([10], dtype=object),
        best_pv=np.asarray(["e2e4 e7e5"], dtype=object),
        subpar_pv=np.asarray(["d2d4 d7d5"], dtype=object),
        metadata=np.asarray(json.dumps(metadata), dtype=object),
    )

    concept_dir = tmp_path / "concept"
    concept_dir.mkdir()
    (concept_dir / "report.json").write_text(
        json.dumps(
            {
                "method": "dynamic_sparse_cvxpy",
                "status": "optimal",
                "mode": "flat",
                "index_mode": "both",
                "reverse": False,
                "pairs": "pairs.train.npz",
                "num_pairs": 1,
                "dimension": 8,
                "norm": 0.5,
                "constraint_satisfaction": 1.0,
                "margin_satisfaction": 1.0,
                "objective": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (concept_dir / "novelty_report.json").write_text(
        json.dumps(
            {
                "machine_samples": 7,
                "human_samples": 10,
                "accepted_vectors": [0],
                "vectors": [
                    {
                        "vector": 0,
                        "novelty_area": 0.125,
                        "positive_rank_fraction": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (concept_dir / "baselines_report.json").write_text(
        json.dumps(
            {
                "nonzero_features": 1,
                "actual": {
                    "constraint_satisfaction": 1.0,
                    "mean_score": 2.0,
                },
                "random_sparse": {
                    "constraint_satisfaction_mean": 0.25,
                },
                "shuffled_labels": {
                    "constraint_satisfaction_mean": 0.5,
                },
                "shuffled_solve": {
                    "count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (concept_dir / "heldout_eval_report.json").write_text(
        json.dumps(
            {
                "split": "test",
                "num_pairs": 3,
                "dimension": 8,
                "direction_key": "raw_direction",
                "nonzero_features": 1,
                "evaluation": {
                    "constraint_satisfaction": 0.75,
                    "margin_satisfaction": 0.5,
                    "mean_score": 1.25,
                    "min_score": -0.25,
                },
            }
        ),
        encoding="utf-8",
    )
    (concept_dir / "prototypes_report.json").write_text(
        json.dumps(
            {
                "split": "train",
                "direction_key": "direction",
                "score_summary": {"max": 3.0, "mean": 1.5},
                "prototypes": [
                    {
                        "index": 4,
                        "score": 3.0,
                        "best_moves": "e2e4",
                        "subpar_moves": "d2d4",
                    }
                ],
                "random_controls": [{"index": 0, "score": 0.0}],
            }
        ),
        encoding="utf-8",
    )
    (concept_dir / "policy_margin_report.json").write_text(
        json.dumps(
            {
                "num_pairs": 1,
                "layer": "trunk",
                "alpha": 1.0,
                "mean_base_margin": 0.5,
                "mean_patched_margin": 0.75,
                "mean_delta_margin": 0.25,
                "fraction_delta_positive": 1.0,
                "top1_change_rate": 0.0,
                "top1_legal_masked": True,
            }
        ),
        encoding="utf-8",
    )

    report = build_dynamic_concept_report(
        pairs_path=pairs,
        concept_dir=concept_dir,
        top_n=1,
    )

    assert "# Dynamic Concept Report" in report
    assert "- status: optimal" in report
    assert "- solver pairs: `pairs.train.npz`" in report
    assert f"- report pairs: `{pairs}`" in report
    assert "- activation key: token_activations" in report
    assert "- accepted vectors: [0]" in report
    assert "- vector 0: novelty_area=0.125000" in report
    assert "- actual constraint satisfaction: 1.000000" in report
    assert "- random sparse constraint satisfaction mean: 0.250000" in report
    assert "- split: test" in report
    assert "- direction key: raw_direction" in report
    assert "- constraint satisfaction: 0.750000" in report
    assert "- margin satisfaction: 0.500000" in report
    assert "- selected prototypes: 1" in report
    assert "- top prototype: index=4, score=3.000000, best=e2e4, subpar=d2d4" in report
    assert "- mean delta margin: 0.250000" in report
    assert "- top1 change rate: 0.000000" in report
    assert "- top1 legal masked: True" in report
    assert "| 0 | e2e4 (50) | d2d4 (10) | 40 | e2e4 e7e5 | d2d4 d7d5 | root fen |" in report


def test_build_dynamic_concept_report_includes_family_summaries(tmp_path):
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 8), dtype=np.float32),
        root_fens=np.asarray(["root fen"], dtype=object),
    )
    concept_dir = tmp_path / "concept"
    concept_dir.mkdir()
    (concept_dir / "report.json").write_text(
        json.dumps(
            {
                "method": "dynamic_sparse_cvxpy",
                "status": "optimal",
                "pairs": "pairs.train.npz",
                "num_pairs": 1,
                "dimension": 8,
            }
        ),
        encoding="utf-8",
    )

    families_dir = tmp_path / "dynamic_families"
    family_dir = families_dir / "family_000"
    family_dir.mkdir(parents=True)
    (families_dir / "report.json").write_text(
        json.dumps(
            {
                "method": "dynamic_concept_families",
                "clusters_requested": 2,
                "families_solved": 1,
                "bootstrap_count": 3,
                "skipped_clusters": [
                    {"cluster_id": 1, "num_rows": 1, "reason": "below_min_cluster_size"}
                ],
                "families": [
                    {
                        "family_id": 0,
                        "cluster_id": 0,
                        "num_pairs": 6,
                        "status": "optimal",
                        "constraint_satisfaction": 1.0,
                        "margin_satisfaction": 0.6,
                        "objective": 12.5,
                        "screening": {
                            "enabled": True,
                            "method": "abs_mean",
                            "screened_dimension": 4,
                            "max_features": 8,
                        },
                        "stability": {"min_cosine": 0.75, "pass_fraction": 0.667},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (family_dir / "heldout_eval_report.json").write_text(
        json.dumps(
            {
                "evaluation": {
                    "constraint_satisfaction": 0.8,
                    "margin_satisfaction": 0.4,
                    "mean_score": 1.2,
                }
            }
        ),
        encoding="utf-8",
    )
    (family_dir / "prototypes_report.json").write_text(
        json.dumps(
            {
                "score_summary": {"max": 2.0},
                "prototypes": [{"index": 1}, {"index": 2}],
            }
        ),
        encoding="utf-8",
    )
    (family_dir / "policy_margin_report.json").write_text(
        json.dumps(
            {
                "mean_delta_margin": 0.125,
                "fraction_delta_positive": 0.5,
            }
        ),
        encoding="utf-8",
    )

    report = build_dynamic_concept_report(
        pairs_path=pairs,
        concept_dir=concept_dir,
        families_path=families_dir / "report.json",
        top_n=0,
    )

    assert "## Concept Families" in report
    assert "- clusters requested: 2" in report
    assert "- families solved: 1" in report
    assert "- skipped clusters: 1" in report
    assert (
        "| 0 | optimal | 0 | 6 | 1.000000 | 0.600000 | 12.500000 | "
        "0.800000 | 0.400000 | 0.750000 | 0.667000 | abs_mean/4/8 | "
        "2 | 2.000000 | 0.125000 | 0.500000 |"
    ) in report


def test_build_dynamic_concept_report_finds_default_families_report(tmp_path):
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 8), dtype=np.float32),
        root_fens=np.asarray(["root fen"], dtype=object),
    )
    concept_dir = tmp_path / "concept"
    concept_dir.mkdir()
    (concept_dir / "report.json").write_text(
        json.dumps(
            {
                "method": "dynamic_sparse_cvxpy",
                "status": "optimal",
                "num_pairs": 1,
                "dimension": 8,
            }
        ),
        encoding="utf-8",
    )
    (concept_dir / "families_report.json").write_text(
        json.dumps(
            {
                "method": "dynamic_concept_families",
                "clusters_requested": 1,
                "families_solved": 1,
                "bootstrap_count": 0,
                "skipped_clusters": [],
                "families": [
                    {
                        "family_id": 0,
                        "cluster_id": 0,
                        "num_pairs": 3,
                        "status": "optimal_inaccurate",
                        "constraint_satisfaction": 0.75,
                        "margin_satisfaction": 0.25,
                        "objective": 1.5,
                        "screening": {"enabled": False},
                        "stability": {"enabled": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_dynamic_concept_report(
        pairs_path=pairs,
        concept_dir=concept_dir,
        top_n=0,
    )

    assert "- families solved: 1" in report
    assert (
        "| 0 | optimal_inaccurate | 0 | 3 | 0.750000 | 0.250000 | "
        "1.500000 | n/a | n/a | n/a | n/a | disabled | n/a | n/a | n/a | n/a |"
    ) in report


def test_build_dynamic_concept_report_accepts_explicit_evaluation_path(tmp_path):
    pairs = tmp_path / "pairs.test.npz"
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 8), dtype=np.float32),
        root_fens=np.asarray(["root fen"], dtype=object),
    )
    concept_dir = tmp_path / "concept"
    concept_dir.mkdir()
    (concept_dir / "report.json").write_text(
        json.dumps(
            {
                "method": "dynamic_sparse_cvxpy",
                "status": "optimal",
                "pairs": "pairs.train.npz",
                "num_pairs": 10,
                "dimension": 8,
            }
        ),
        encoding="utf-8",
    )
    explicit_eval = tmp_path / "heldout_eval_report.json"
    explicit_eval.write_text(
        json.dumps(
            {
                "split": "custom",
                "num_pairs": 1,
                "dimension": 8,
                "direction_key": "raw_direction",
                "nonzero_features": 2,
                "evaluation": {
                    "constraint_satisfaction": 1.0,
                    "margin_satisfaction": 1.0,
                    "mean_score": 3.0,
                    "min_score": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_dynamic_concept_report(
        pairs_path=pairs,
        concept_dir=concept_dir,
        evaluation_path=explicit_eval,
        top_n=0,
    )

    assert "- solver pairs: `pairs.train.npz`" in report
    assert f"- report pairs: `{pairs}`" in report
    assert "- split: custom" in report
    assert "- mean score: 3.000000" in report


def test_build_dynamic_concept_report_cli_accepts_explicit_evaluation_path(
    tmp_path,
    monkeypatch,
):
    pairs = tmp_path / "pairs.test.npz"
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 8), dtype=np.float32),
        root_fens=np.asarray(["root fen"], dtype=object),
    )
    concept_dir = tmp_path / "concept"
    concept_dir.mkdir()
    (concept_dir / "report.json").write_text(
        json.dumps(
            {
                "method": "dynamic_sparse_cvxpy",
                "status": "optimal",
                "pairs": "pairs.train.npz",
                "num_pairs": 10,
                "dimension": 8,
            }
        ),
        encoding="utf-8",
    )
    explicit_eval = tmp_path / "explicit_eval.json"
    explicit_eval.write_text(
        json.dumps(
            {
                "split": "cli-test",
                "num_pairs": 1,
                "dimension": 8,
                "direction_key": "raw_direction",
                "nonzero_features": 2,
                "evaluation": {
                    "constraint_satisfaction": 1.0,
                    "margin_satisfaction": 1.0,
                    "mean_score": 4.0,
                    "min_score": 4.0,
                },
            }
        ),
        encoding="utf-8",
    )
    explicit_prototypes = tmp_path / "explicit_prototypes.json"
    explicit_prototypes.write_text(
        json.dumps(
            {
                "split": "cli-train",
                "direction_key": "direction",
                "reverse": True,
                "score_summary": {"max": 5.0, "mean": 2.5},
                "prototypes": [
                    {
                        "index": 2,
                        "score": 5.0,
                        "best_moves": "g1f3",
                        "subpar_moves": "b1c3",
                    }
                ],
                "random_controls": [],
            }
        ),
        encoding="utf-8",
    )
    families_dir = tmp_path / "families"
    families_dir.mkdir()
    explicit_families = families_dir / "report.json"
    explicit_families.write_text(
        json.dumps(
            {
                "method": "dynamic_concept_families",
                "clusters_requested": 1,
                "families_solved": 1,
                "bootstrap_count": 0,
                "skipped_clusters": [],
                "families": [
                    {
                        "family_id": 0,
                        "cluster_id": 0,
                        "num_pairs": 4,
                        "status": "optimal",
                        "constraint_satisfaction": 0.5,
                        "screening": {"enabled": False},
                        "stability": {"enabled": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_dynamic_concept_report.py",
            "--pairs",
            str(pairs),
            "--concept",
            str(concept_dir),
            "--evaluation",
            str(explicit_eval),
            "--prototypes",
            str(explicit_prototypes),
            "--families",
            str(explicit_families),
            "--out",
            str(out),
            "--top-n",
            "0",
        ],
    )

    assert build_dynamic_concept_report_cli.main() == 0

    report = out.read_text(encoding="utf-8")
    assert "- solver pairs: `pairs.train.npz`" in report
    assert f"- report pairs: `{pairs}`" in report
    assert "- split: cli-test" in report
    assert "- mean score: 4.000000" in report
    assert "- reverse: True" in report
    assert "- top prototype: index=2, score=5.000000, best=g1f3, subpar=b1c3" in report
    assert "- clusters requested: 1" in report
    assert (
        "| 0 | optimal | 0 | 4 | 0.500000 | n/a | n/a | n/a | n/a | n/a | "
        "n/a | disabled | n/a | n/a | n/a | n/a |"
    ) in report


def test_build_dynamic_concept_report_handles_partial_pair_metadata(tmp_path):
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 8), dtype=np.float32),
        root_fens=np.asarray(["root fen"], dtype=object),
    )
    concept_dir = tmp_path / "concept"
    concept_dir.mkdir()
    (concept_dir / "report.json").write_text(
        json.dumps(
            {
                "method": "dynamic_sparse_cvxpy",
                "status": "optimal",
                "num_pairs": 1,
                "dimension": 8,
            }
        ),
        encoding="utf-8",
    )

    report = build_dynamic_concept_report(
        pairs_path=pairs,
        concept_dir=concept_dir,
        top_n=1,
    )

    assert "| 0 |  (None) |  (None) | n/a |  |  | root fen |" in report


def test_build_dynamic_concept_report_rejects_negative_top_n(tmp_path):
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 8), dtype=np.float32),
    )
    concept_dir = tmp_path / "concept"
    concept_dir.mkdir()
    (concept_dir / "report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="top_n must be >= 0"):
        build_dynamic_concept_report(
            pairs_path=pairs,
            concept_dir=concept_dir,
            top_n=-1,
        )
