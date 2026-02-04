"""Constrained NanoChat GPT implementation.

This module is the supported import location used by tests:

    from CausalGPT.constrained_nanochat import GPT, GPTConfig

The implementation currently lives in `CausalGPT/backup/constrained_nanochat.py`.
We re-export symbols here to keep a stable public import path.
"""

from __future__ import annotations

from CausalGPT.backup.constrained_nanochat import (  # noqa: F401
    GPT,
    GPTConfig,
    _normalize_anc_text,
)

__all__ = [
    "GPT",
    "GPTConfig",
    "_normalize_anc_text",
]
