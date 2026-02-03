from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from CausalGPT.utils.dnn_constraints_utils import parse_anc_file, train_dnn_with_constraints

degree = 5  # 可根据实际调整

def _causalgpt_root() -> Path:
    # .../src/CausalGPT/tests/test_model1.py -> .../src/CausalGPT
    return Path(__file__).resolve().parents[1]


def _cfg_path() -> Path:
    return _causalgpt_root() / "configs" / "exdbn_perform.yaml"


def _ensure_minimal_anc_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if path.read_text().strip():
                return
        except Exception:
            pass
    path.write_text("arcs{\n}\n")


def test_yaml_paths_integration_runs_fast():
    cfg_file = _cfg_path()
    assert cfg_file.exists(), f"Missing YAML config: {cfg_file}"

    cfg = OmegaConf.load(cfg_file)
    data_path = Path(cfg.data_path)
    anc_path = Path(cfg.anc_path)

    assert data_path.exists(), f"Missing data_path from YAML: {data_path}"
    _ensure_minimal_anc_file(anc_path)

    X = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    n, d = X.shape
    assert n > 0 and d > 0

    mask = parse_anc_file(str(anc_path), d)
    assert mask.shape == (d, d)

    # Keep it fast for tests
    adj_dnn = train_dnn_with_constraints(d, mask, epochs=2)
    assert adj_dnn.shape == (d, d)