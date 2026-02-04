from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request


@dataclass(frozen=True)
class LLMPriorConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_edges: int = 30
    timeout_seconds: int = 120


class LLMPriorError(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM output.

    Accepts either pure JSON or text that contains a single JSON object.
    """
    text = text.strip()
    if not text:
        raise LLMPriorError("Empty LLM response")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to extract the first {...} block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise LLMPriorError("LLM response does not contain a JSON object")

    try:
        obj = json.loads(match.group(0))
    except Exception as e:
        raise LLMPriorError(f"Failed to parse extracted JSON: {e}") from e

    if not isinstance(obj, dict):
        raise LLMPriorError("Extracted JSON is not an object")
    return obj


def _normalize_pairs(value: Any, *, n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if value is None:
        return pairs
    if not isinstance(value, list):
        return pairs

    for item in value:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], (int, float, str))
            and isinstance(item[1], (int, float, str))
        ):
            try:
                i = int(item[0])
                j = int(item[1])
            except Exception:
                continue
            if i == j:
                continue
            if 0 <= i < n and 0 <= j < n:
                pairs.append((i, j))

    # de-dup while preserving order
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _openai_compat_chat_completions(
    *,
    cfg: LLMPriorConfig,
    messages: list[dict[str, str]],
) -> str:
    base = cfg.base_url.rstrip("/")
    url = f"{base}/v1/chat/completions"

    api_key = os.getenv(cfg.api_key_env)
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": float(cfg.temperature),
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise LLMPriorError(f"LLM request failed: {e}") from e

    try:
        obj = json.loads(raw)
        content = obj["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("content is not a string")
        return content
    except Exception as e:
        raise LLMPriorError(f"Unexpected LLM response schema: {e}") from e


def build_llm_prior_prompt(
    *,
    variables: list[str],
    dataset_hint: str | None,
    max_edges: int,
) -> list[dict[str, str]]:
    # We ask for indices only to avoid name-matching ambiguity.
    var_lines = [f"{i}: {name}" for i, name in enumerate(variables)]
    var_block = "\n".join(var_lines)
    dataset_hint = dataset_hint or "(unknown dataset)"

    system = (
        "You are a causal discovery assistant. "
        "Given variable names only, propose a small set of directed causal relations "
        "as soft priors. Output STRICT JSON only. No markdown, no prose."
    )

    user = (
        f"Dataset: {dataset_hint}\n\n"
        "Variables (index: name):\n"
        f"{var_block}\n\n"
        "Task:\n"
        f"- Propose at most {int(max_edges)} directed relations as ancestor priors (soft).\n"
        "- Return indices only; do not invent new variables.\n"
        "- Prefer obvious directions (e.g., age -> disease) and avoid cycles when possible.\n\n"
        "Output JSON schema (exact keys):\n"
        "{\n"
        "  \"anc\": [[src_idx, dst_idx], ...],\n"
        "  \"forb_anc\": [[src_idx, dst_idx], ...],\n"
        "  \"notes\": \"optional short string\"\n"
        "}\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_or_query_llm_priors(
    *,
    variables: list[str],
    dataset_hint: str | None,
    cfg: LLMPriorConfig,
    cache_path: Path,
    refresh: bool = False,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, Any]]:
    """Return (anc, forb_anc, meta). Uses cache when available."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not refresh:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            anc = _normalize_pairs(cached.get("anc"), n=len(variables))
            forb_anc = _normalize_pairs(cached.get("forb_anc"), n=len(variables))
            return anc, forb_anc, {"source": "cache", "cache_path": str(cache_path)}
        except Exception:
            # fall through to re-query
            pass

    if cfg.provider.lower() in {"none", "off", "disabled"}:
        raise LLMPriorError("LLM priors requested but provider is disabled")

    messages = build_llm_prior_prompt(
        variables=variables,
        dataset_hint=dataset_hint,
        max_edges=cfg.max_edges,
    )

    if cfg.provider.lower() in {"openai_compat", "openai-compatible", "ollama"}:
        content = _openai_compat_chat_completions(cfg=cfg, messages=messages)
    else:
        raise LLMPriorError(f"Unknown provider: {cfg.provider}")

    obj = _extract_json_object(content)
    anc = _normalize_pairs(obj.get("anc"), n=len(variables))
    forb_anc = _normalize_pairs(obj.get("forb_anc"), n=len(variables))

    payload = {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "dataset_hint": dataset_hint,
        "variables": variables,
        "anc": anc,
        "forb_anc": forb_anc,
        "notes": obj.get("notes", ""),
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return anc, forb_anc, {"source": "llm", "cache_path": str(cache_path)}
