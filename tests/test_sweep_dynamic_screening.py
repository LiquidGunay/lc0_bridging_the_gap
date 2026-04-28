import json

import numpy as np

from tools import sweep_dynamic_screening


def _write_pairs(path, differences):
    row_count = differences.shape[0]
    np.savez_compressed(
        path,
        differences=differences.astype(np.float32),
        root_fens=np.asarray(["8/8/8/8/8/8/4P3/K6k w - - 0 1"] * row_count, dtype=object),
        best_moves=np.asarray(["e2e3"] * row_count, dtype=object),
        subpar_moves=np.asarray(["e2e4"] * row_count, dtype=object),
    )


def test_sweep_dynamic_screening_cli_writes_summary_and_artifacts(tmp_path, monkeypatch):
    train = tmp_path / "pairs.train.npz"
    test = tmp_path / "pairs.test.npz"
    train_differences = np.tile(np.asarray([0.0, 2.0, 0.0, 0.0]), (6, 1))
    test_differences = np.asarray([
        [0.0, 2.0, 0.0, 0.0],
        [0.0, -2.0, 0.0, 0.0],
    ])
    _write_pairs(train, train_differences)
    _write_pairs(test, test_differences)
    out = tmp_path / "sweep"

    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_dynamic_screening.py",
            "--train-pairs",
            str(train),
            "--test-pairs",
            str(test),
            "--out",
            str(out),
            "--max-features",
            "1,2",
            "--screening-methods",
            "abs_mean,mean_abs",
            "--random-count",
            "2",
            "--shuffled-label-count",
            "2",
            "--top-k",
            "1",
            "--prototype-random-count",
            "1",
            "--skip-policy-margin",
            "--no-standardize",
        ],
    )

    assert sweep_dynamic_screening.main() == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["method"] == "dynamic_screening_sweep"
    assert len(summary["rows"]) == 4
    assert len(summary["configs"]) == 4
    assert summary["policy_alphas"] == []

    first = summary["rows"][0]
    assert first["solve_status"] == "optimal"
    assert first["heldout_constraint_satisfaction"] == 0.5
    assert first["curriculum_lines"] == 2
    assert first["largest_abs_policy_mean_delta_margin"] is None
    assert first["best_positive_policy_mean_delta_margin"] is None

    concept_dir = out / "abs_mean_1"
    assert (concept_dir / "report.json").exists()
    assert (concept_dir / "heldout_eval_report.json").exists()
    assert (concept_dir / "baselines_report.json").exists()
    assert (concept_dir / "prototypes_report.json").exists()
    assert (concept_dir / "teachability_curriculum.jsonl").exists()
    direction = np.load(concept_dir / "concept_direction.npz", allow_pickle=True)
    assert direction["direction"].shape == (4,)
    np.testing.assert_array_equal(direction["screening_indices"], np.asarray([1]))
    assert "abs_mean_1" in (out / "summary.md").read_text(encoding="utf-8")


def test_sweep_dynamic_screening_reverse_flips_training_objective(tmp_path, monkeypatch):
    train = tmp_path / "pairs.train.npz"
    test = tmp_path / "pairs.test.npz"
    differences = np.tile(np.asarray([0.0, -2.0, 0.0, 0.0]), (6, 1))
    _write_pairs(train, differences)
    _write_pairs(test, differences[:2])
    out = tmp_path / "sweep"

    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_dynamic_screening.py",
            "--train-pairs",
            str(train),
            "--test-pairs",
            str(test),
            "--out",
            str(out),
            "--max-features",
            "1",
            "--screening-methods",
            "abs_mean",
            "--random-count",
            "0",
            "--shuffled-label-count",
            "0",
            "--top-k",
            "1",
            "--prototype-random-count",
            "0",
            "--skip-policy-margin",
            "--no-standardize",
            "--reverse",
        ],
    )

    assert sweep_dynamic_screening.main() == 0

    report = json.loads((out / "abs_mean_1" / "report.json").read_text(encoding="utf-8"))
    assert report["reverse"] is True
    assert report["constraint_satisfaction"] == 1.0
    direction = np.load(out / "abs_mean_1" / "concept_direction.npz", allow_pickle=True)
    assert direction["raw_direction"][1] > 0.0
