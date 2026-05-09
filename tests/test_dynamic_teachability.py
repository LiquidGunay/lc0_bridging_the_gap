import json

import numpy as np
import pytest

from lc0jax.interpretability.dynamic_teachability import (
    curriculum_pair_indices,
    evaluate_policy_adapter,
    teachability_curriculum_records,
    teachability_lift_report,
    train_low_rank_policy_adapter,
)
from tools import evaluate_teachability, export_teachability_curriculum


def _prototypes_report():
    return {
        "split": "train",
        "direction_key": "direction",
        "reverse": True,
        "pairs": "pairs.train.npz",
        "concept": "concept_dir",
        "seed": 3,
        "score_mode": "auto",
        "prototypes": [
            {
                "rank": 0,
                "index": 7,
                "score": 3.0,
                "projection_score": -3.0,
                "root_fens": "root-a",
                "best_moves": "e2e4",
                "subpar_moves": "d2d4",
                "source_ids": "game-a",
                "policy_logits": [1.0, 0.0],
            }
        ],
        "random_controls": [
            {
                "rank": 0,
                "index": 2,
                "score": 0.25,
                "projection_score": -0.25,
                "root_fens": "root-b",
                "best_moves": "g1f3",
                "subpar_moves": "b1c3",
                "source_ids": "game-b",
            }
        ],
    }


def test_teachability_curriculum_records_exports_prototypes_and_controls():
    records = teachability_curriculum_records(_prototypes_report())

    assert [record["group"] for record in records] == ["prototype", "random_control"]
    assert records[0]["pair_index"] == 7
    assert records[0]["target_move"] == "e2e4"
    assert records[0]["contrast_move"] == "d2d4"
    assert records[0]["reverse"] is True
    assert records[0]["provenance"] == {
        "pairs": "pairs.train.npz",
        "concept": "concept_dir",
        "seed": 3,
        "score_mode": "auto",
    }
    assert records[0]["metadata"]["source_ids"] == "game-a"
    assert "policy_logits" not in records[0]["metadata"]
    assert records[1]["target_move"] == "g1f3"


def test_teachability_curriculum_records_can_limit_groups():
    records = teachability_curriculum_records(
        _prototypes_report(),
        max_prototypes=1,
        max_controls=0,
    )

    assert len(records) == 1
    assert records[0]["group"] == "prototype"


def test_teachability_curriculum_records_rejects_negative_limits():
    with pytest.raises(ValueError, match="max_prototypes"):
        teachability_curriculum_records(_prototypes_report(), max_prototypes=-1)
    with pytest.raises(ValueError, match="max_controls"):
        teachability_curriculum_records(_prototypes_report(), max_controls=-1)


def test_teachability_curriculum_records_rejects_malformed_rows():
    report = _prototypes_report()
    del report["prototypes"][0]["best_moves"]

    with pytest.raises(ValueError, match="best_moves"):
        teachability_curriculum_records(report)


def test_export_teachability_curriculum_cli_writes_jsonl(tmp_path, monkeypatch):
    prototypes = tmp_path / "prototypes_report.json"
    prototypes.write_text(json.dumps(_prototypes_report()), encoding="utf-8")
    out = tmp_path / "curriculum.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_teachability_curriculum.py",
            "--prototypes",
            str(prototypes),
            "--out",
            str(out),
            "--max-prototypes",
            "1",
            "--max-controls",
            "1",
        ],
    )

    assert export_teachability_curriculum.main() == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["group"] for row in rows] == ["prototype", "random_control"]
    assert rows[0]["root_fen"] == "root-a"
    assert rows[0]["provenance"]["pairs"] == "pairs.train.npz"
    assert rows[1]["metadata"]["source_ids"] == "game-b"


def test_export_teachability_curriculum_cli_rejects_negative_limits(tmp_path, monkeypatch):
    prototypes = tmp_path / "prototypes_report.json"
    prototypes.write_text(json.dumps(_prototypes_report()), encoding="utf-8")
    out = tmp_path / "curriculum.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_teachability_curriculum.py",
            "--prototypes",
            str(prototypes),
            "--out",
            str(out),
            "--max-prototypes",
            "-1",
        ],
    )

    with pytest.raises(ValueError, match="max_prototypes"):
        export_teachability_curriculum.main()


def _toy_features_and_teacher_logits():
    features = np.asarray(
        [
            [2.0, 0.0],
            [1.5, 0.2],
            [-2.0, 0.0],
            [-1.5, -0.2],
            [1.2, 0.1],
            [-1.2, -0.1],
        ],
        dtype=np.float32,
    )
    teacher_logits = np.asarray(
        [
            [4.0, -4.0],
            [3.0, -3.0],
            [-4.0, 4.0],
            [-3.0, 3.0],
            [2.5, -2.5],
            [-2.5, 2.5],
        ],
        dtype=np.float32,
    )
    return features, teacher_logits


def test_train_low_rank_policy_adapter_fits_toy_teacher_logits():
    features, teacher_logits = _toy_features_and_teacher_logits()

    trained = train_low_rank_policy_adapter(
        features,
        teacher_logits,
        train_indices=np.arange(features.shape[0]),
        hidden_dim=2,
        steps=80,
        batch_size=4,
        learning_rate=0.2,
        seed=11,
    )
    metrics = evaluate_policy_adapter(
        trained["params"],
        features,
        teacher_logits,
        indices=np.arange(features.shape[0]),
    )

    assert trained["history"]["final_full_loss"] < trained["history"]["initial_loss"]
    assert "last_minibatch_loss" in trained["history"]
    assert metrics["top1_overlap"] >= 0.8
    assert metrics["kl_teacher_student"] < 0.5


def test_teachability_lift_report_compares_prototypes_and_controls():
    features, teacher_logits = _toy_features_and_teacher_logits()
    records = [
        {"group": "prototype", "pair_index": 0},
        {"group": "prototype", "pair_index": 2},
        {"group": "random_control", "pair_index": 1},
        {"group": "random_control", "pair_index": 3},
    ]

    report = teachability_lift_report(
        features,
        teacher_logits,
        records,
        eval_features=features[4:],
        eval_teacher_logits=teacher_logits[4:],
        hidden_dim=2,
        steps=50,
        batch_size=2,
        learning_rate=0.2,
        seed=13,
    )

    assert curriculum_pair_indices(records, group="prototype").tolist() == [0, 2]
    assert report["method"] == "low_rank_policy_adapter_kl"
    assert report["prototype"]["available"] is True
    assert report["random_control"]["available"] is True
    assert report["eval_num_rows"] == 2
    assert report["prototype"]["eval"]["rows"] == 2
    assert report["lift"]["available"] is True
    assert report["lift"]["budget_matched"] is True
    assert "top1_overlap_lift" in report["lift"]


def test_teachability_lift_report_marks_mismatched_budgets_unavailable():
    features, teacher_logits = _toy_features_and_teacher_logits()
    records = [
        {"group": "prototype", "pair_index": 0},
        {"group": "prototype", "pair_index": 2},
        {"group": "random_control", "pair_index": 1},
    ]

    report = teachability_lift_report(
        features,
        teacher_logits,
        records,
        eval_features=features[4:],
        eval_teacher_logits=teacher_logits[4:],
        hidden_dim=2,
        steps=10,
        batch_size=2,
        learning_rate=0.2,
        seed=13,
    )

    assert report["prototype"]["available"] is True
    assert report["random_control"]["available"] is True
    assert report["lift"]["available"] is False
    assert report["lift"]["reason"] == "train_row_budget_mismatch"
    assert report["lift"]["prototype_train_rows"] == 2
    assert report["lift"]["random_control_train_rows"] == 1


def test_evaluate_teachability_cli_writes_report(tmp_path, monkeypatch):
    features, teacher_logits = _toy_features_and_teacher_logits()
    data = tmp_path / "teachability_data.npz"
    np.savez_compressed(data, differences=features, policy_logits=teacher_logits)
    eval_data = tmp_path / "teachability_eval_data.npz"
    np.savez_compressed(
        eval_data,
        differences=features[4:],
        policy_logits=teacher_logits[4:],
    )
    curriculum = tmp_path / "curriculum.jsonl"
    curriculum.write_text(
        "\n".join(
            [
                json.dumps({"group": "prototype", "pair_index": 0}),
                json.dumps({"group": "prototype", "pair_index": 2}),
                json.dumps({"group": "random_control", "pair_index": 1}),
                json.dumps({"group": "random_control", "pair_index": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "teachability_report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_teachability.py",
            "--features",
            str(data),
            "--eval-features",
            str(eval_data),
            "--curriculum",
            str(curriculum),
            "--out",
            str(out),
            "--hidden-dim",
            "2",
            "--steps",
            "30",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.2",
        ],
    )

    assert evaluate_teachability.main() == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["features"] == str(data)
    assert report["eval_features"] == str(eval_data)
    assert report["curriculum"] == str(curriculum)
    assert report["eval_rows"] == 2
    assert report["prototype"]["train_rows"] == 2
    assert report["random_control"]["train_rows"] == 2
