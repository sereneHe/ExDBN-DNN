import pytest
from pathlib import Path
from CausalGPT.constrained_nanochat import _normalize_anc_text, GPT, GPTConfig
import torch

def test_constrained_nanochat_with_exdbn_arcs():
    anc_path = Path("/Users/xiaoyuhe/EXDBN-LLM/ExDBN-DNN/reports/causalgpt_runs_codiet/codiet/codiet_302_0_conf0.99999/ExDBN_LLM.anc")
    anc_text = anc_path.read_text()
    norm_text, arcs = _normalize_anc_text(anc_text)
    # 构造一个简单的 mask 或输入，假设节点名全集
    node_names = sorted(set([src for src, dst, _ in arcs] + [dst for src, dst, _ in arcs]))
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    n = len(node_names)
    # 生成一个 mask，1 表示允许，0 表示禁止
    mask = torch.ones((n, n), dtype=torch.float32)
    for src, dst, _ in arcs:
        i, j = name_to_idx[src], name_to_idx[dst]
        mask[i, j] = 0  # 例如将 arcs 作为禁止边
    # 构造模型
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=32)
    model = GPT(config)
    model.init_weights()
    # 这里 mask 可作为 loss/约束输入，具体用法视你的训练/推理流程而定
    print("节点名:", node_names)
    print("mask shape:", mask.shape)
    print("mask 禁止边数:", int((mask==0).sum().item()))
    assert mask.shape[0] == mask.shape[1] == len(node_names)
