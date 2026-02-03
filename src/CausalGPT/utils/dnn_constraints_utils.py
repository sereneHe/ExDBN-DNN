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

    We interpret lines with "A -> B" as forbidden directed edges A->B.
    (This is a very lightweight parser intended for the constraints files
    produced in this repo.)
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


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim: int, *, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, *, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = int(n_head)
        self.n_embd = int(n_embd)
        self.dropout = float(dropout)

        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=bias)
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self._flash = hasattr(F, "scaled_dot_product_attention")
        if not self._flash:
            mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = c // self.n_head

        q = q.view(b, t, self.n_head, head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_dim).transpose(1, 2)

        if self._flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, *, n_embd: int, dropout: float, bias: bool):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, *, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class NanoGPTConfig:
    n_nodes: int
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True


class NanoGPTAdjacency(nn.Module):
    """A tiny nanoGPT-style decoder that outputs adjacency logits.

    Nodes are tokens 0..n-1. For each token position i, output a length-n
    vector of logits for edges i -> j.
    """

    def __init__(self, cfg: NanoGPTConfig):
        super().__init__()
        self.cfg = cfg
        n = int(cfg.n_nodes)

        self.wte = nn.Embedding(n, cfg.n_embd)
        self.wpe = nn.Embedding(n, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=cfg.n_embd,
                    n_head=cfg.n_head,
                    block_size=n,
                    dropout=cfg.dropout,
                    bias=cfg.bias,
                )
                for _ in range(int(cfg.n_layer))
            ]
        )
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, n, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) with T == n_nodes
        b, t = idx.shape
        if t != int(self.cfg.n_nodes):
            raise ValueError(f"Expected sequence length {self.cfg.n_nodes}, got {t}")

        pos = torch.arange(0, t, device=idx.device, dtype=torch.long)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, n_nodes)


def train_dnn_with_constraints(
    n: int,
    forbidden_mask: np.ndarray,
    *,
    target_adj: np.ndarray | None = None,
    epochs: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Train a nanoGPT-style model to reproduce target adjacency under forbidden-edge constraints."""

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

    cfg = NanoGPTConfig(n_nodes=int(n))
    model = NanoGPTAdjacency(cfg).to(device)
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
