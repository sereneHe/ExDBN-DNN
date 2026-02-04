import torch

from CausalGPT.utils.read_anc_convert_token import generate_causal_batch


def test_exdbn_anc_to_causal_batch() -> None:
    """Minimal, self-contained test for anc-driven token priors.

    - Ensures banned tokens are never generated.
    - Ensures replace priors can deterministically steer generation.
    """

    # Small synthetic ANC with both a banned edge and a replace prior.
    # Use high confidence so the replacement gets a large positive weight.
    anc_text = """arcs{
  A -> B 0.99999;
  A => C 0.99999;
}
"""

    # Build a tiny vocab that covers all tokens referenced in anc_text.
    vocab_dict = {
        "PAD": 0,
        "B": 1,
        "A": 2,
        "C": 3,
    }

    class DummyModel(torch.nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            self.vocab_size = int(vocab_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Return uniform logits (all ones), shape: (B, T, V)
            b, t = x.shape
            return torch.ones((b, t, self.vocab_size), dtype=torch.float32, device=x.device)

    model = DummyModel(vocab_size=max(vocab_dict.values()) + 1)

    # Seed tokens include token_id1 (A) so causal_probs tracking can trigger.
    tokens_batch = [[vocab_dict["A"]], [vocab_dict["A"]]]

    generated, causal_probs = generate_causal_batch(
        model,
        tokens_batch,
        vocab_dict,
        anc_text,
        max_tokens=3,
        temperature=0.0,  # deterministic argmax
        device="cpu",
    )

    banned_id = vocab_dict["B"]
    replace_target_id = vocab_dict["C"]

    # Banned token must never appear.
    assert all(tok != banned_id for seq in generated for tok in seq)
    # Replace prior should push C to be the argmax.
    assert all(tok == replace_target_id for seq in generated for tok in seq)

    # Check causal_probs records the replace prior once A is in the history.
    key = f"{vocab_dict['A']}->{vocab_dict['C']}"
    assert all(key in d for d in causal_probs)
