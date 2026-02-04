import pytest
from pathlib import Path
from CausalGPT.constrained_nanochat import _normalize_anc_text
from CausalGPT.model import train_dnn_with_constraints
import numpy as np

def test_end2end_exdbn_anc_to_dnn():
    anc_path = Path("/Users/xiaoyuhe/EXDBN-LLM/ExDBN-DNN/reports/causalgpt_runs_codiet/codiet/codiet_302_0_conf0.99999/ExDBN_LLM.anc")
    anc_text = anc_path.read_text()
    norm_text, arcs = _normalize_anc_text(anc_text)
    node_names = sorted(set([src for src, dst, _ in arcs] + [dst for src, dst, _ in arcs]))
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    n = len(node_names)
    mask = np.ones((n, n), dtype=int)
    for src, dst, _ in arcs:
        i, j = name_to_idx[src], name_to_idx[dst]
        mask[i, j] = 0  # 禁止边
    # 直接用 mask 作为 DNN 结构约束
    adj = train_dnn_with_constraints(n, mask, epochs=2)  # epochs可调
    print("DNN结构预测：\n", adj)
    assert adj.shape == (n, n)
