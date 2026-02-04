import pytest
import torch
from CausalGPT.constrained_nanochat import GPT, GPTConfig

def test_constrained_nanochat_exdbn_smoke():
    # Minimal config for fast smoke test (ve_gate_channels=32, n_embd=32 to match)
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=32)
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, config.vocab_size, (1, config.sequence_len))
    logits = model(x)
    assert logits.shape == (1, config.sequence_len, config.vocab_size)
    tokens = x[0].tolist()[:2]
    out = list(model.generate(tokens, max_tokens=4, temperature=0.5, top_k=5))
    assert len(out) <= 4

def test_constrained_nanochat_exdbn_full_feature():
    # Full-feature config to cover all features (more layers, heads, embedding dim)
    config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)
    model = GPT(config)
    model.init_weights()
    x = torch.randint(0, config.vocab_size, (1, config.sequence_len))
    logits = model(x)
    assert logits.shape == (1, config.sequence_len, config.vocab_size)
    tokens = x[0].tolist()[:4]
    out = list(model.generate(tokens, max_tokens=8, temperature=1.0, top_k=8))
    assert len(out) <= 8
