import json

import numpy as np

from lc0jax.interpretability.dynamic_families import solve_dynamic_concept_families
from tools import solve_dynamic_concept_families as family_cli


def _two_family_differences() -> np.ndarray:
    left = np.tile(np.asarray([2.0, 0.0, 0.0, 0.0]), (6, 1))
    right = np.tile(np.asarray([0.0, 3.0, 0.0, 0.0]), (6, 1))
    return np.vstack([left, right])


def test_solve_dynamic_concept_families_clusters_and_bootstraps():
    result = solve_dynamic_concept_families(
        _two_family_differences(),
        n_clusters=2,
        min_cluster_size=3,
        max_features=1,
        standardize=False,
        seed=3,
        bootstrap_count=2,
        bootstrap_min_rows=3,
    )

    assert result["labels"].shape == (12,)
    assert len(result["families"]) == 2
    np.testing.assert_array_equal(result["family_ids"], np.asarray([0, 1]))
    assert result["cluster_ids"].shape == (2,)
    assert result["directions"].shape == (2, 4)
    assert {family["num_rows"] for family in result["families"]} == {6}
    assert set(np.argmax(np.abs(result["directions"]), axis=1).tolist()) == {0, 1}
    for family in result["families"]:
        assert family["constraint_satisfaction"] == 1.0
        assert family["stability"]["completed"] == 2
        assert family["stability"]["min_cosine"] > 0.99


def test_solve_dynamic_concept_families_cli_writes_family_artifacts(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(pairs, differences=_two_family_differences())
    out = tmp_path / "families"

    monkeypatch.setattr(
        "sys.argv",
        [
            "solve_dynamic_concept_families.py",
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--clusters",
            "2",
            "--min-cluster-size",
            "3",
            "--max-features",
            "1",
            "--no-standardize",
            "--bootstrap-count",
            "1",
            "--bootstrap-min-rows",
            "3",
        ],
    )

    assert family_cli.main() == 0

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["method"] == "dynamic_concept_families"
    assert report["families_solved"] == 2
    assert report["clusters_requested"] == 2
    assert report["families"][0]["stability"]["completed"] == 1
    assert (out / "families.npz").exists()
    families_npz = np.load(out / "families.npz", allow_pickle=True)
    np.testing.assert_array_equal(families_npz["family_ids"], np.asarray([0, 1]))
    assert families_npz["cluster_ids"].shape == (2,)
    assert (out / "family_000" / "concept_direction.npz").exists()
    family_direction = np.load(out / "family_000" / "concept_direction.npz")
    assert family_direction["direction"].shape == (4,)
