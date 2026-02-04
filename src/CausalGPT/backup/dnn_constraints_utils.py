from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


def parse_anc_file(anc_path: str, n: int) -> np.ndarray:
    """Parse a CaMML-style .anc file into a forbidden-edge mask.

    We interpret lines with "i -> j" as forbidden directed edges i->j.
    This is a lightweight parser intended for the constraints files produced in this repo.

    Returns:
        mask: (n, n) int array with 1 meaning forbidden, 0 meaning allowed.
    """

    mask = np.zeros((n, n), dtype=int)
    with open(anc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "->" not in line:
                continue
            parts = line.strip().replace(";", "").replace("}", "").split()
            if len(parts) < 3:
                continue
            try:
                i, j = int(parts[0]), int(parts[2])
            except Exception:
                continue
            if 0 <= i < n and 0 <= j < n:
                mask[i, j] = 1
    return mask


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Functional RMSNorm with no learnable parameters."""

    return F.rms_norm(x, (x.size(-1),))


def _precompute_rotary_embeddings(
    seq_len: int,
    head_dim: int,
    *,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) rotary caches.

    Shapes:
      cos, sin: (1, T, 1, head_dim/2)
    """

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings")

    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()

    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)

    return cos[None, :, None, :], sin[None, :, None, :]


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings.

    Args:
        x: (B, T, H, D)
        cos/sin: (1, T, 1, D/2)
    """

    if x.ndim != 4:
        raise ValueError("Expected x.ndim == 4")

    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


@dataclass
class NanoChatConfig:
    n_nodes: int
    n_layer: int = 2
    n_head: int = 2
    n_kv_head: int = 2  # GQA: key/value heads
    n_embd: int = 64
    dropout: float = 0.0
    rotary_base: float = 10_000.0


class CausalSelfAttention(nn.Module):
    """Rotary + QK norm + GQA-ready causal self-attention.

    Uses PyTorch SDPA (scaled_dot_product_attention). On supported GPUs it will
    use FlashAttention automatically; on CPU/MPS it falls back appropriately.
    """

    def __init__(self, config: NanoChatConfig):
        super().__init__()

        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.n_kv_head > config.n_head or (config.n_head % config.n_kv_head != 0):
            raise ValueError("Require n_kv_head <= n_head and n_head % n_kv_head == 0")

        self.n_head = int(config.n_head)
        self.n_kv_head = int(config.n_kv_head)
        self.n_embd = int(config.n_embd)
        self.dropout = float(config.dropout)
        self.head_dim = self.n_embd // self.n_head

        # bias-free linear layers
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()

        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_kv_head, self.head_dim)

        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # QK norm
        q = rms_norm(q)
        k = rms_norm(k)

        # Expand KV heads for GQA if needed
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        # SDPA expects (B, H, T, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.permute(0, 2, 1, 3).contiguous().view(b, t, self.n_embd)
        return self.c_proj(y)


class MLP(nn.Module):
    """Bias-free MLP with ReLU^2 activation."""

    def __init__(self, config: NanoChatConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: NanoChatConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), cos, sin)
        x = x + self.mlp(rms_norm(x))
        return x


class NanoChatAdjacency(nn.Module):
    """Nanochat-style GPT that outputs adjacency logits.

    Nodes are tokens 0..n-1. For each token position i, output a length-n vector of logits for edges i -> j.
    """

    def __init__(self, cfg: NanoChatConfig):
        super().__init__()
        self.cfg = cfg
        n = int(cfg.n_nodes)

        self.wte = nn.Embedding(n, cfg.n_embd)
        self.drop = nn.Dropout(float(cfg.dropout))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(int(cfg.n_layer))])
        self.head = nn.Linear(cfg.n_embd, n, bias=False)

        # simple stable init
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        if t != int(self.cfg.n_nodes):
            raise ValueError(f"Expected sequence length {self.cfg.n_nodes}, got {t}")

        # rotary cache for current length
        head_dim = int(self.cfg.n_embd) // int(self.cfg.n_head)
        rot_dtype = torch.bfloat16 if idx.device.type == "cuda" else torch.float32
        cos, sin = _precompute_rotary_embeddings(
            seq_len=t,
            head_dim=head_dim,
            base=float(self.cfg.rotary_base),
            device=idx.device,
            dtype=rot_dtype,
        )

        x = self.wte(idx)
        x = rms_norm(x)  # norm after token embedding
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = rms_norm(x)
        return self.head(x)  # (B, T, n_nodes)


def train_dnn_with_constraints(
    n: int,
    forbidden_mask: np.ndarray,
    *,
    target_adj: np.ndarray | None = None,
    epochs: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Train a transformer to reproduce target adjacency under forbidden-edge constraints.

    - `forbidden_mask[i, j] == 1` means edge i->j is forbidden.
    - We train only on allowed, off-diagonal edges.
    - The target is binary (edge present or not).

    Returns:
        pred: (n, n) int adjacency matrix.
    """

    if forbidden_mask.shape != (n, n):
        raise ValueError(f"forbidden_mask must be shape {(n, n)}, got {forbidden_mask.shape}")

    if target_adj is None:
        target_adj = np.zeros((n, n), dtype=int)
    if target_adj.shape != (n, n):
        raise ValueError(f"target_adj must be shape {(n, n)}, got {target_adj.shape}")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        else "cpu"
    )

    cfg = NanoChatConfig(
        n_nodes=int(n),
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        dropout=0.0,
    )
    model = NanoChatAdjacency(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    forbidden = torch.tensor(forbidden_mask, dtype=torch.float32, device=device)
    allowed = 1.0 - forbidden
    allowed.fill_diagonal_(0.0)

    target = torch.tensor((target_adj != 0).astype(np.float32), dtype=torch.float32, device=device)
    target = target * allowed

    idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(0)  # (1, n)

    for _epoch in range(int(epochs)):
        logits = model(idx).squeeze(0)  # (n, n)
        loss_matrix = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        denom = allowed.sum().clamp(min=1.0)
        loss = (loss_matrix * allowed).sum() / denom

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(idx).squeeze(0)
        probs = torch.sigmoid(logits) * allowed
        pred = (probs > 0.5).to(torch.int64)

    return pred.detach().cpu().numpy().astype(int)
