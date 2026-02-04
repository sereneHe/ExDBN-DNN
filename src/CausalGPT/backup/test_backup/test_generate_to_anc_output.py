import tempfile
from pathlib import Path
import torch
from CausalGPT.constrained_nanochat import GPT, GPTConfig

def test_generate_to_anc_outputs(tmp_path):
    config = GPTConfig(sequence_len=8, vocab_size=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=32)
    model = GPT(config)
    model.init_weights()
    # Dummy vocab and decode
    vocab = [str(i) for i in range(config.vocab_size)]
    def decode(tokens):
        return ' '.join(vocab[t] for t in tokens)
    prompt_tokens = [1, 2, 3]
    out_anc_path = tmp_path / "test_anc.anc"
    node_names = [f"N{i}" for i in range(5)]
    res = model.generate_to_anc(
        prompt_tokens,
        max_tokens=5,
        decode=decode,
        out_anc_path=out_anc_path,
        node_names=node_names,
        default_weight="0.1",
        temperature=0.5,
        top_k=5,
        seed=42,
    )
    print("\n===== ANC FILE CONTENT =====\n")
    print(Path(res["anc_path"]).read_text())
    if res["anc_idx_path"]:
        print("\n===== ANC_IDX FILE CONTENT =====\n")
        print(Path(res["anc_idx_path"]).read_text())
