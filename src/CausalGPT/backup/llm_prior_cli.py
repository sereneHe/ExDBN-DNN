from __future__ import annotations

import argparse
from pathlib import Path


def _sniff_delimiter(path: Path, *, default: str = ",") -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                commas = line.count(",")
                semis = line.count(";")
                if semis > commas:
                    return ";"
                if commas > 0:
                    return ","
                return default
    except Exception:
        return default


def _read_header(path: Path) -> list[str]:
    delim = _sniff_delimiter(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip("\n\r")
    # naive split is fine for typical CSV headers in this repo
    cols = [c.strip().strip('"') for c in header.split(delim)]
    cols = [c for c in cols if c]
    if not cols:
        raise RuntimeError(f"Failed to read header from {path}")
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate soft causal priors from variable names using an LLM.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV file path (uses first row as header).")
    parser.add_argument("--dataset_hint", type=str, default=None, help="Optional dataset hint shown to the LLM.")

    parser.add_argument("--provider", type=str, default="none", help="none | openai_compat | ollama")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com", help="For Ollama: http://localhost:11434")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_edges", type=int, default=30)
    parser.add_argument("--refresh", action="store_true")

    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Where to cache JSON (default: <csv>.llm_prior.json)",
    )
    parser.add_argument(
        "--write_anc",
        type=Path,
        default=None,
        help="If set, write a CaMML .anc file with variable-name labels.",
    )
    parser.add_argument("--conf", type=float, default=0.99999)

    args = parser.parse_args()

    variables = _read_header(args.csv)
    dataset_hint = args.dataset_hint or args.csv.stem
    cache_path = args.cache or args.csv.with_suffix(args.csv.suffix + ".llm_prior.json")

    from CausalGPT.utils.llm_prior import LLMPriorConfig, load_or_query_llm_priors

    cfg = LLMPriorConfig(
        provider=str(args.provider),
        model=str(args.model),
        base_url=str(args.base_url),
        api_key_env=str(args.api_key_env),
        max_edges=int(args.max_edges),
    )

    anc, forb_anc, meta = load_or_query_llm_priors(
        variables=variables,
        dataset_hint=dataset_hint,
        cfg=cfg,
        cache_path=cache_path,
        refresh=bool(args.refresh),
    )

    print(f"[INFO] Loaded LLM priors: anc={len(anc)}, forb_anc={len(forb_anc)}")
    print(f"[INFO] Cache: {meta.get('cache_path')}")

    # Print args you can pass to ExDBN_perform.py
    print("[OUT] --ancs=\"" + str(anc).replace(" ", "") + "\"")
    print("[OUT] --forb_ancs=\"" + str(forb_anc).replace(" ", "") + "\"")

    if args.write_anc is not None:
        from CausalGPT import exdbn_ban_edges

        exdbn_ban_edges.write_camml_anc(
            args.write_anc,
            n_nodes=len(variables),
            conf=float(args.conf),
            anc=list(anc),
            forb_anc=list(forb_anc),
            abs_edges=None,
            labels=variables,
        )
        print(f"[OUT] wrote .anc: {args.write_anc}")


if __name__ == "__main__":
    main()
