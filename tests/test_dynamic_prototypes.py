import json

import numpy as np

from lc0jax.interpretability.dynamic_prototypes import (
    dynamic_prototype_report,
    projection_scores,
    select_random_indices,
    select_top_indices,
)
from tools import select_dynamic_prototypes


def test_dynamic_prototype_report_selects_top_rows_and_random_controls():
    differences = np.asarray(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [-1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    direction = np.asarray([1.0, 0.0])
    metadata = {
        "root_fens": np.asarray(["root0", "root1", "root2", "root3"], dtype=object),
        "best_moves": np.asarray(["a", "b", "c", "d"], dtype=object),
        "subpar_moves": np.asarray(["aa", "bb", "cc", "dd"], dtype=object),
    }

    scores = projection_scores(differences, direction)
    np.testing.assert_allclose(scores, [1.0, 3.0, -1.0, 2.0])
    np.testing.assert_array_equal(select_top_indices(scores, top_k=2), [1, 3])
    random_indices = select_random_indices(num_rows=4, count=2, seed=0, exclude=np.asarray([1, 3]))
    assert set(random_indices.tolist()).issubset({0, 2})

    report = dynamic_prototype_report(
        differences,
        direction,
        metadata,
        top_k=2,
        random_count=2,
        seed=0,
        split_name="train",
    )

    assert report["method"] == "dynamic_prototype_selection"
    assert report["top_k"] == 2
    assert report["random_count"] == 2
    assert [row["index"] for row in report["prototypes"]] == [1, 3]
    assert report["prototypes"][0]["root_fens"] == "root1"
    assert report["prototypes"][0]["best_moves"] == "b"
    assert {row["index"] for row in report["random_controls"]}.isdisjoint({1, 3})


def test_select_dynamic_prototypes_cli_writes_report(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.train.npz"
    np.savez_compressed(
        pairs,
        differences=np.asarray([[1.0, 0.0], [4.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        root_fens=np.asarray(["root0", "root1", "root2"], dtype=object),
        best_moves=np.asarray(["a", "b", "c"], dtype=object),
        subpar_moves=np.asarray(["aa", "bb", "cc"], dtype=object),
    )
    concept = tmp_path / "concept"
    concept.mkdir()
    np.savez_compressed(
        concept / "concept_direction.npz",
        direction=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    out = tmp_path / "prototypes_report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "select_dynamic_prototypes.py",
            "--pairs",
            str(pairs),
            "--concept",
            str(concept),
            "--out",
            str(out),
            "--top-k",
            "2",
            "--random-count",
            "1",
            "--split-name",
            "train",
        ],
    )

    assert select_dynamic_prototypes.main() == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["pairs"] == str(pairs)
    assert report["concept"] == str(concept)
    assert report["split"] == "train"
    assert [row["index"] for row in report["prototypes"]] == [1, 2]
    assert report["random_controls"][0]["index"] == 0
