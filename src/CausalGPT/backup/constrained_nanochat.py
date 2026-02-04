"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from CausalGPT.utils.anc_arcs import parse_anc_arcs


_ANC_ARC_RE = re.compile(
    r"(?P<src>[A-Za-z0-9_]+)\s*->\s*(?P<dst>[A-Za-z0-9_]+)(?:\s+(?P<w>[-+0-9.eE]+))?\s*;?"
)


def _normalize_anc_text(raw_text: str, *, default_weight: str = "0.00001") -> tuple[str, list[tuple[str, str, str]]]:
    """Extract arcs from raw model text and rebuild a minimal CaMML-like anc block.

    Output format:
        arcs{\n
        SRC -> DST WEIGHT;\n
        }\n

    Returns:
        (anc_text, arcs) where arcs is a list of (src, dst, weight_str).
    """

    arcs: list[tuple[str, str, str]] = []
    for m in _ANC_ARC_RE.finditer(raw_text):
        src = m.group("src")
        dst = m.group("dst")
        w = m.group("w") or default_weight
        arcs.append((src, dst, w))

    lines = ["arcs{"]
    for src, dst, w in arcs:
        lines.append(f"{src} -> {dst} {w};")
    lines.append("}")
    return "\n".join(lines) + "\n", arcs


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# Optional nanochat dependency:
# This repo vendors a nanochat-style GPT implementation, but the upstream nanochat package
# is not always installed in the runtime environment. For EXDBN integration and anc
# generation, we only need the model forward/generate path, so we provide minimal
# fallbacks when nanochat is unavailable.
try:
    from nanochat.utils import pos_impl_constraint
    from nanochat.common import get_dist_info, print0
    from nanochat.muon import Muon, DistMuon
    from nanochat.adamw import DistAdamW
    from nanochat.ssl_alm_adam import SSLALM_Adam
    from nanochat.PBM import PBM

    # Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
    from nanochat.flash_attention import flash_attn
except ModuleNotFoundError:
    pos_impl_constraint = None

    def get_dist_info():
        return 0, 1

    def print0(*args, **kwargs):
        print(*args, **kwargs)

    Muon = DistMuon = DistAdamW = SSLALM_Adam = PBM = None

    class _FlashAttnCompat:
        @staticmethod
        def flash_attn_func(q, k, v, causal=True, window_size=None):
            # q,k,v are (B, T, H, D). SDPA expects (B, H, T, D).
            q2 = q.permute(0, 2, 1, 3)
            k2 = k.permute(0, 2, 1, 3)
            v2 = v.permute(0, 2, 1, 3)
            y = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=None, is_causal=bool(causal))
            return y.permute(0, 2, 1, 3)

        @staticmethod
        def flash_attn_with_kvcache(*args, **kwargs):
            raise NotImplementedError("KV-cache path requires the nanochat flash_attention module")

    flash_attn = _FlashAttnCompat

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
        # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale),  # same LR as token embedding
            dict(params=resid_params, lr=scalar_lr * 0.01), # these are a lot more sensitive because they accumulate in the residual stream
            dict(params=x0_params, lr=scalar_lr),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers


    def setup_constrained_optimizer(self, optimizer, scalar_lr=0.5, m=1, device='cuda', opt_kwargs=None, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95)):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)
        # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale * 2),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale * 2),
            dict(params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 2),  # same LR as token embedding
            dict(params=resid_params, lr=scalar_lr * 0.01 * 2), # these are a lot more sensitive because they accumulate in the residual stream
            dict(params=x0_params, lr=scalar_lr * 2),
        ]
        
        if optimizer == 'alm':
            opt_kwargs = opt_kwargs | dict(beta1=adam_betas[0], beta2=adam_betas[1], eps=1e-10, dual_lr=0.5, dual_bound=500, device=device)
            optimizer = SSLALM_Adam(groups, m=m, **opt_kwargs)
        else:
            opt_kwargs = opt_kwargs | dict(betas=adam_betas, eps=1e-10, device=device, mu=0, init_dual=5.0, dual_range=(1e-2, 1e+2))
            optimizer = PBM(groups, m=m, **opt_kwargs)
        optimizers = [optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, eval_constraints=False, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if not eval_constraints:
                # unconstrained training: return just loss
                return loss
            else:
                # constrained training: return loss and constraints
                batch_probs = torch.nn.functional.softmax(logits, dim=-1)
                batch_con = self.calculate_replace_token_penalty_batch(batch_probs, 10, 32)
                # batch_con = self.calculate_banned_token_penalty_batch(batch_probs, 10) # '\n' = 10
                return loss, batch_con
        else:
            # inference: just return the logits directly
            return logits

    def calculate_replace_token_penalty_batch(self, prob_batch, token_id1, token_id2):
        # "instead of predicting token_id1, predict token_id2"
        return torch.mean(
            torch.sum(prob_batch[:,:, token_id1] - prob_batch[:,:, token_id2], dim=1)
        )

    def calculate_banned_token_penalty_batch(self, prob_batch, token_id):
        """
        Return the mean predicted probability of a token over the entire batch.
        
        :param prob_batch: a batch of sequences of probability vectors.
        :param token_id: token index in the probability vectors.
        """
        # "do not predict token_id"
        return torch.mean(
            torch.sum(prob_batch[:,:,token_id], dim=1)
        )

    def calculate_unclosed_bracket_penalty(self, prob_sequence):
        """
        Calculate a penalty for each unclosed bracket based on the model's predicted probabilities.

        Args:
            prob_sequence (list(tensor)): List of tensors probabilities corresponding to each token.

        Returns:
            Tensor: Total penalty for unclosed brackets.
        """

        # "for each predicted token 40 - open bracket - predict token 41 - closed bracket - somewhere later"

        # Supports either a list[Tensor(vocab)] or a Tensor(T, vocab)
        if isinstance(prob_sequence, torch.Tensor):
            probs_list = [prob_sequence[i] for i in range(prob_sequence.size(0))]
        else:
            probs_list = list(prob_sequence)

        if len(probs_list) == 0:
            return torch.tensor(0.0)

        # ) = 41, ( = 40, ] = 93, [ = 91, } = 125, { = 123
        # For .anc files, curly braces are important; we include all three common bracket types.
        bracket_pairs = {41: 40, 93: 91, 125: 123}
        open_brackets = set(bracket_pairs.values())
        close_brackets = set(bracket_pairs.keys())

        stack: list[int] = []
        open_positions: list[tuple[int, int]] = []  # (pos, open_token_id)

        for pos, probs in enumerate(probs_list):
            token = int(torch.argmax(probs).detach().cpu().item())
            if token in open_brackets:
                stack.append(token)
                open_positions.append((pos, token))
            elif token in close_brackets:
                expected_open = bracket_pairs[token]
                if stack and stack[-1] == expected_open:
                    stack.pop()
                    # remove the most recent matching open bracket
                    for k in range(len(open_positions) - 1, -1, -1):
                        if open_positions[k][1] == expected_open:
                            open_positions.pop(k)
                            break

        penalty = probs_list[0].new_tensor(0.0)
        for pos, open_tok in open_positions:
            penalty = penalty + probs_list[pos][open_tok]
        return penalty


    @torch.inference_mode()
    def generate_to_anc(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        *,
        decode: Callable[[list[int]], str],
        out_anc_path: str | Path,
        node_names: list[str] | None = None,
        default_weight: str = "0.00001",
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
        stop_token_ids: tuple[int, ...] = (125,),
        priors_anc_text: str | None = None,
        priors_vocab_dict: dict[str, int] | None = None,
    ) -> dict:
        """Generate an anc file and return artifacts for downstream connection processing.

        This is intended to connect the model's token generation with the existing pipeline
        that expects a `.anc` (and preferably an `_idx.anc`) file.

        Args:
            prompt_tokens: token ids used as prompt.
            max_tokens: maximum number of newly generated tokens.
            decode: function that converts token ids into text.
            out_anc_path: where to write the named `.anc` file.
            node_names: if provided, also writes an `_idx.anc` by mapping names -> indices.
            stop_token_ids: if any of these token ids are generated, stop early.

        Returns:
            dict with keys: anc_path, anc_idx_path (optional), forbidden_mask (optional), anc_text.
        """

        generated: list[int] = []
        for tok in self.generate(
            prompt_tokens,
            max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            anc_text=priors_anc_text,
            vocab_dict=priors_vocab_dict,
        ):
            generated.append(int(tok))
            if stop_token_ids and int(tok) in stop_token_ids:
                break

        full_tokens = list(prompt_tokens) + generated
        prompt_text = decode(list(prompt_tokens))
        generated_text = decode(list(generated))
        raw_text = decode(full_tokens)
        anc_text, arcs = _normalize_anc_text(raw_text, default_weight=default_weight)

        out_anc_path = Path(out_anc_path)
        _write_text(out_anc_path, anc_text)

        anc_idx_path: Path | None = None
        forbidden_mask = None
        if node_names is not None:
            name_to_idx = {name: i for i, name in enumerate(node_names)}
            idx_lines = ["arcs{"]
            for src, dst, w in arcs:
                if src in name_to_idx and dst in name_to_idx:
                    idx_lines.append(f"{name_to_idx[src]} -> {name_to_idx[dst]} {w};")
            idx_lines.append("}")
            anc_idx_text = "\n".join(idx_lines) + "\n"
            anc_idx_path = out_anc_path.with_name(out_anc_path.stem + "_idx" + out_anc_path.suffix)
            _write_text(anc_idx_path, anc_idx_text)

            try:
                from CausalGPT.utils.dnn_constraints_utils import parse_anc_file

                forbidden_mask = parse_anc_file(str(anc_idx_path), len(node_names))
            except Exception:
                forbidden_mask = None

        return {
            "anc_path": str(out_anc_path),
            "anc_idx_path": str(anc_idx_path) if anc_idx_path is not None else None,
            "forbidden_mask": forbidden_mask,
            "anc_text": anc_text,
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "raw_text": raw_text,
        }



    @torch.inference_mode()
    def generate(
        self,
        tokens,
        max_tokens,
        temperature=1.0,
        top_k=None,
        seed=42,
        *,
        anc_text: str | None = None,
        vocab_dict: dict[str, int] | None = None,
    ):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()

        token_priors = None
        if anc_text is not None:
            if vocab_dict is None:
                raise ValueError("vocab_dict is required when anc_text is provided")
            token_priors = parse_anc_arcs(anc_text, vocab_dict)

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)

            # Apply .anc priors (if provided)
            if token_priors:
                for tp in token_priors:
                    if tp.get("type") == "banned":
                        logits[:, int(tp["token_id"])] = -float("inf")
                for tp in token_priors:
                    if tp.get("type") == "replace":
                        w = tp.get("weight", 0.0)
                        if isinstance(w, torch.Tensor):
                            w = w.to(device=logits.device, dtype=logits.dtype)
                        logits[:, int(tp["token_id2"])] += w
                        logits[:, int(tp["token_id1"])] -= w

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token