import json

import numpy as np

from tools import solve_dynamic_concepts


def test_solve_dynamic_concepts_cli_writes_screening_metadata(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(
        pairs,
        differences=np.tile(np.asarray([0.0, 2.0, 0.0, 0.0], dtype=np.float32), (6, 1)),
    )
    out = tmp_path / "concept"

    monkeypatch.setattr(
        "sys.argv",
        [
            "solve_dynamic_concepts.py",
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--max-features",
            "1",
            "--no-standardize",
        ],
    )

    assert solve_dynamic_concepts.main() == 0

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["method"] == "dynamic_screened_sparse_cvxpy"
    assert report["dimension"] == 4
    assert report["screening"]["enabled"] is True
    assert report["screening"]["screened_dimension"] == 1
    assert report["screening"]["selected_feature_preview"] == [1]

    direction = np.load(out / "concept_direction.npz", allow_pickle=True)
    assert direction["direction"].shape == (4,)
    np.testing.assert_array_equal(direction["screening_indices"], np.asarray([1]))
