import json

import pytest

from lc0jax.interpretability.dynamic_teachability import teachability_curriculum_records
from tools import export_teachability_curriculum


def _prototypes_report():
    return {
        "split": "train",
        "direction_key": "direction",
        "reverse": True,
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
    assert records[0]["metadata"]["source_ids"] == "game-a"
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
    assert rows[1]["metadata"]["source_ids"] == "game-b"
