import re
import torch
import torch.nn.functional as F

# =============================
# 1️⃣ 解析 anc 文件
# =============================

def anc_prob_to_weight(p, eps=1e-6):
    p = torch.clamp(torch.tensor(p, dtype=torch.float32), eps, 1-eps)
    return torch.log(p / (1 - p))


def parse_anc_arcs(anc_text, vocab_dict):
    pattern = r"(\w+)\s*(->|=>)\s*(\w+)\s*([\d.]+);"
    matches = re.findall(pattern, anc_text)
    token_priors = []
    for v1, arc_type, v2, prob_str in matches:
        token_id1 = vocab_dict[v1]
        token_id2 = vocab_dict[v2]
        anc_prob = float(prob_str)
        weight = anc_prob_to_weight(anc_prob)
        if arc_type == "->":
            token_priors.append({"type":"banned","token_id":token_id2,"weight":weight})
        elif arc_type == "=>":
            token_priors.append({"type":"replace","token_id1":token_id1,"token_id2":token_id2,"weight":weight})
    return token_priors


# =============================
# 2️⃣ Token-level prior loss
# =============================

def anc_prior_loss_batch(prob_batch, token_priors):
    B, T, V = prob_batch.shape
    loss = torch.zeros(1, device=prob_batch.device)
    for tp in token_priors:
        if tp["type"] == "banned":
            token_prob = prob_batch[:,:,tp["token_id"]]
            loss += (tp["weight"] * token_prob).mean()
        elif tp["type"] == "replace":
            prob1 = prob_batch[:,:,tp["token_id1"]]
            prob2 = prob_batch[:,:,tp["token_id2"]]
            loss += (tp["weight"] * (prob1 - prob2)).mean()
    return loss


# =============================
# 3️⃣ 训练阶段整合示例
# =============================

def training_step(model, tokens, targets, token_priors, optimizer):
    logits = model(tokens)  # (B,T,V)
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    probs = F.softmax(logits, dim=-1)
    prior_loss = anc_prior_loss_batch(probs, token_priors)
    total_loss = ce_loss + prior_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), ce_loss.item(), prior_loss.item()


# =============================
# 4️⃣ 生成阶段使用 priors
# =============================

@torch.inference_mode()
def generate_causal_batch(model, tokens_batch, vocab_dict, anc_text,
                          max_tokens=10, temperature=1.0, top_k=None, device=None):
    if device is None:
        device = next(model.parameters()).device

    B = len(tokens_batch)
    max_len = max(len(seq) for seq in tokens_batch)
    ids = torch.full((B, max_len), 0, dtype=torch.long, device=device)
    for i, seq in enumerate(tokens_batch):
        ids[i,:len(seq)] = torch.tensor(seq, device=device)

    token_priors = parse_anc_arcs(anc_text, vocab_dict)
    generated = [[] for _ in range(B)]
    causal_probs = [{} for _ in range(B)]

    for _ in range(max_tokens):
        logits = model(ids)[:, -1, :]

        for tp in token_priors:
            if tp["type"] == "banned":
                logits[:, tp["token_id"]] = -float("inf")
            elif tp["type"] == "replace":
                logits[:, tp["token_id2"]] += tp["weight"]
                logits[:, tp["token_id1"]] -= tp["weight"]

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits[logits < v[:, [-1]]] = -float('inf')

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)
        else:
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.argmax(probs, dim=-1, keepdim=True)

        ids = torch.cat([ids, next_ids], dim=1)
        for b in range(B):
            generated[b].append(next_ids[b].item())
            for tp in token_priors:
                if tp["type"] == "replace":
                    k = f"{tp['token_id1']}->{tp['token_id2']}"
                    if k not in causal_probs[b]:
                        if tp['token_id1'] in ids[b,:-1].tolist():
                            causal_probs[b][k] = probs[b, tp['token_id2']].item()

    return generated, causal_probs


# =============================
# 5️⃣ 使用示例
# =============================
if __name__ == "__main__":
    # 假设 vocab_dict
    vocab_dict = {"A1MHMS":10, "A1MHMU":11, "A3MHMS":12}
    anc_text = """
arcs{
  A1MHMS -> A1MHMU 0.99999;
  A1MHMS => A3MHMS 0.00001;
}
"""

    token_priors = parse_anc_arcs(anc_text, vocab_dict)

    # batch 数据示例
    tokens_batch = [[10],[10]]
    max_tokens = 5

    # 假设 model 和 optimizer 已定义
    # training_step(model, tokens, targets, token_priors, optimizer)

    # 生成示例
    # generated, causal_probs = generate_causal_batch(model, tokens_batch, vocab_dict, anc_text, max_tokens)
    # print(generated, causal_probs)