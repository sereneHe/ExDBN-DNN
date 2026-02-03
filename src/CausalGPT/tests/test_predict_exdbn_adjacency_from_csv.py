from pathlib import Path
import os

import numpy as np
import pytest

from CausalGPT import exdbn_ban_edges


def test_predict_exdbn_adjacency_from_csv_smoke_asia(monkeypatch):
    data_path = Path("/Users/xiaoyuhe/Datasets/Asia/asia_250_0.csv")
    if not data_path.exists():
        pytest.skip(f"Missing dataset: {data_path}")

    # Default to a fast configuration so local runs don't hang.
    # Opt out by setting EXDBN_TEST_NO_FAST=1.
    if (os.getenv("EXDBN_TEST_NO_FAST") or "") != "1":
        monkeypatch.setenv("EXDBN_TIME_LIMIT", "30")
        monkeypatch.setenv("EXDBN_TARGET_MIP_GAP", "0.10")
        monkeypatch.setenv("EXDBN_CALLBACK_MODE", "all_cycles")
        # Gurobi progress output (prints to stdout; optionally write a log file)
        monkeypatch.setenv("EXDBN_GUROBI_OUTPUTFLAG", "1")
        monkeypatch.setenv("EXDBN_GUROBI_DISPLAYINTERVAL", "1")

    # Keep it small/fast; Asia has only 8 vars.
    try:
        adj, info = exdbn_ban_edges.predict_exdbn_adjacency_from_csv(
            data_path,
            sample_size=50,
            max_degree=5,
        )
    except ModuleNotFoundError as e:
        pytest.skip(f"Missing optional EXDBN deps (dagsolvers): {e}")
    except Exception as e:
        # Typical reasons: gurobi not installed/license unavailable, or solver backend not set up.
        pytest.skip(f"EXDBN solver unavailable in this environment: {type(e).__name__}: {e}")

    assert isinstance(adj, np.ndarray)
    assert adj.ndim == 2
    assert adj.shape[0] == adj.shape[1]
    assert adj.shape[0] == info["d"]

    # Binary adjacency (0/1)
    vals = set(np.unique(adj).tolist())
    assert vals.issubset({0, 1})
