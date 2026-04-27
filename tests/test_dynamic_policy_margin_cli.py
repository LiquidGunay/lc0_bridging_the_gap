import sys

import numpy as np
import pytest

from tools import dynamic_policy_margin


def test_load_pair_rows_skips_illegal_moves_in_cli_validation(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(
        pairs,
        differences=np.ones((1, 2), dtype=np.float32),
        root_fens=np.asarray(["8/8/8/8/8/8/8/K6k w - - 0 1"], dtype=object),
        best_moves=np.asarray(["e2e4"], dtype=object),
        subpar_moves=np.asarray(["d2d4"], dtype=object),
    )
    concept = tmp_path / "concept"
    concept.mkdir()
    np.savez_compressed(concept / "concept_direction.npz", direction=np.ones(2))

    monkeypatch.setattr(dynamic_policy_margin, "load_pb_gz", lambda _path: object())
    monkeypatch.setattr(dynamic_policy_margin, "map_bt4_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sys, "argv", [
        "dynamic_policy_margin.py",
        "--pairs",
        str(pairs),
        "--concept",
        str(concept),
        "--pb",
        "fake.pb.gz",
        "--out",
        str(tmp_path / "out.json"),
    ])

    with pytest.raises(ValueError, match="No valid pair rows remained"):
        dynamic_policy_margin.main()
