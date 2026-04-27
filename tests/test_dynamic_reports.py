import json

import numpy as np

from lc0jax.interpretability.dynamic_reports import build_dynamic_concept_report


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

    report = build_dynamic_concept_report(
        pairs_path=pairs,
        concept_dir=concept_dir,
        top_n=1,
    )

    assert "# Dynamic Concept Report" in report
    assert "- status: optimal" in report
    assert "- activation key: token_activations" in report
    assert "- accepted vectors: [0]" in report
    assert "- vector 0: novelty_area=0.125000" in report
    assert "| 0 | e2e4 (50) | d2d4 (10) | 40 | e2e4 e7e5 | d2d4 d7d5 | root fen |" in report
