from __future__ import annotations

from pathlib import Path

import numpy as np

from CausalGPT.utils.dnn_constraints_utils import parse_anc_file, train_dnn_with_constraints


def test_parse_anc_file_empty(tmp_path: Path):
    anc_path = tmp_path / "test_ancs.anc"
    anc_path.write_text("arcs{\n}\n")

    n = 5
    mask = parse_anc_file(str(anc_path), n)
    assert mask.shape == (n, n)
    assert mask.sum() == 0


def test_train_dnn_with_constraints_runs_small(tmp_path: Path):
    anc_path = tmp_path / "test_ancs.anc"
    anc_path.write_text("arcs{\n0 -> 1 0.00001;\n}\n")

    n = 4
    mask = parse_anc_file(str(anc_path), n)
    adj = train_dnn_with_constraints(n, mask, epochs=2)
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (n, n)
    # Forbidden edge should be masked out
    assert adj[0, 1] == 0
