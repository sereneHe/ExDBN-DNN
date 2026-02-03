#生成和Asia数据集的真实因果图对齐的测试代码
from pathlib import Path

import numpy as np

from CausalGPT.utils.asia_ground_truth_utils import (
    load_asia_ground_truth,
    reorder_asia_csv_to_mapping,
)


def test_load_asia_ground_truth_shapes():
    gt = load_asia_ground_truth(
        mapping_path=Path("/Users/xiaoyuhe/EXDBN-LLM/LLM_CD/BN_structure/mappings/asia.mapping"),
        graph_path=Path("/Users/xiaoyuhe/EXDBN-LLM/LLM_CD/BN_structure/asia_graph.txt"),
    )
    assert gt.adjacency.shape == (8, 8)
    assert len(gt.variable_order) == 8
    assert set(np.unique(gt.adjacency)).issubset({0, 1})


def test_reorder_asia_csv_to_mapping_columns():
    df, info = reorder_asia_csv_to_mapping(
        csv_path=Path("/Users/xiaoyuhe/Datasets/Asia/asia_250_0.csv"),
        out_csv_path=None,
        mapping_path=Path("/Users/xiaoyuhe/EXDBN-LLM/LLM_CD/BN_structure/mappings/asia.mapping"),
    )
    assert list(df.columns) == info["mapping_order"]
    assert df.shape[1] == 8
