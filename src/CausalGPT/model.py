@torch.inference_mode()
def generate_with_priors(self, tokens, max_tokens, vocab_dict, anc_text,
                         temperature=1.0, top_k=None, seed=42):
    """
    Autoregressive generation with .anc priors applied in real-time.
    """
    device = self.get_device()
    rng = None
    if temperature > 0:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

    ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, T)
    token_priors = parse_anc_arcs(anc_text, vocab_dict)

    for _ in range(max_tokens):
        logits = self.forward(ids)[:, -1, :]  # (1, vocab_size)

        # 1️⃣ Apply hard banned tokens
        for tp in token_priors:
            if tp["type"] == "banned":
                logits[:, tp["token_id"]] = -float("inf")

        # 2️⃣ Apply soft replace/prior
        for tp in token_priors:
            if tp["type"] == "replace":
                # "increase probability of token_id2 over token_id1"
                logits[:, tp["token_id2"]] += tp["weight"]
                logits[:, tp["token_id1"]] -= tp["weight"]

        # 3️⃣ Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        # 4️⃣ Sampling
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        else:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)

        ids = torch.cat((ids, next_ids), dim=1)
        yield next_ids.item()
