import json

import numpy as np

from lc0jax.interpretability.dynamic_evaluation import dynamic_evaluation_report
from tools import evaluate_dynamic_concept


def test_dynamic_evaluation_report_scores_direction_on_pairs():
    report = dynamic_evaluation_report(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        np.asarray([1.0, 0.0]),
        margin=0.5,
        split_name="test",
    )

    assert report["method"] == "dynamic_direction_evaluation"
    assert report["split"] == "test"
    assert report["num_pairs"] == 3
    assert report["dimension"] == 2
    assert report["nonzero_features"] == 1
    assert report["evaluation"]["constraint_satisfaction"] == 1 / 3
    assert report["evaluation"]["margin_satisfaction"] == 1 / 3
    assert report["evaluation"]["mean_score"] == 0.0


def test_evaluate_dynamic_concept_cli_writes_report(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.test.npz"
    np.savez_compressed(
        pairs,
        differences=np.asarray([[2.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
    )
    concept = tmp_path / "concept"
    concept.mkdir()
    np.savez_compressed(
        concept / "concept_direction.npz",
        direction=np.asarray([1.0, 0.0], dtype=np.float32),
        raw_direction=np.asarray([0.25, 0.0], dtype=np.float32),
    )
    out = tmp_path / "heldout_eval_report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_dynamic_concept.py",
            "--pairs",
            str(pairs),
            "--concept",
            str(concept),
            "--out",
            str(out),
            "--margin",
            "1.0",
            "--split-name",
            "test",
        ],
    )

    assert evaluate_dynamic_concept.main() == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["pairs"] == str(pairs)
    assert report["concept"] == str(concept)
    assert report["split"] == "test"
    assert report["direction_key"] == "raw_direction"
    assert report["evaluation"]["constraint_satisfaction"] == 0.5
    assert report["evaluation"]["margin_satisfaction"] == 0.0
    assert report["evaluation"]["max_score"] == 0.5
