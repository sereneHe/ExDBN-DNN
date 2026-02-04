"""Helpers for parsing EXDBN-style `.anc` arcs into token-level priors.

Some tests import `parse_anc_arcs` from `CausalGPT.utils.anc_arcs`.
The implementation lives in `CausalGPT.utils.anc_convert_token`.
"""

from __future__ import annotations

from CausalGPT.utils.anc_convert_token import anc_prob_to_weight, parse_anc_arcs

__all__ = [
    "anc_prob_to_weight",
    "parse_anc_arcs",
]
