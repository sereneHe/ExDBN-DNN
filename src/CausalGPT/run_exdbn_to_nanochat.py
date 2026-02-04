from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Callable


def _read_vocab(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty vocab file: {path}")
    return lines


def _make_vocab_dict(vocab: list[str]) -> dict[str, int]:
    return {piece: i for i, piece in enumerate(vocab)}


def _make_decode_from_vocab(vocab: list[str]) -> Callable[[list[int]], str]:
    def decode(ids: list[int]) -> str:
        pieces: list[str] = []
        for i in ids:
            if 0 <= i < len(vocab):
                pieces.append(vocab[i])
        return "".join(pieces)

    return decode


def _load_labels_from_csv(path: Path, *, delimiter: str = ",") -> list[str]:
    try:
        import pandas as pd

        df0 = pd.read_csv(path, sep=delimiter, nrows=0)
        cols = [str(c) for c in df0.columns]
        if not cols:
            raise ValueError("CSV has no header columns")
        return cols
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV header labels from {path}: {e}") from e


def _parse_prompt_tokens(s: str) -> list[int]:
    toks: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        toks.append(int(part))
    if not toks:
        raise ValueError("prompt_tokens is empty")
    return toks


def main() -> None:
    p = argparse.ArgumentParser(description="Run NanoChat using EXDBN-generated .anc as priors")
    p.add_argument("--data_csv", type=Path, required=True)
    p.add_argument("--exdbn_anc", type=Path, required=True, help="Named .anc (ExDBN_LLM.anc) used as priors")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument(
        "--prompt_tokens",
        type=str,
        default="1,2,3",
        help="Comma-separated token ids. Default is a placeholder; set explicitly for real runs.",
    )
    p.add_argument("--delimiter", type=str, default=",")
    p.add_argument("--max_gen", type=int, default=512)
    p.add_argument("--out_anc", type=Path, required=True, help="Output path for generated hard constraints (.anc)")

    args = p.parse_args()

    if not args.data_csv.exists():
        raise SystemExit(f"Missing --data_csv: {args.data_csv}")
    if not args.exdbn_anc.exists():
        raise SystemExit(f"Missing --exdbn_anc: {args.exdbn_anc}")
    if not args.ckpt.exists():
        raise SystemExit(f"Missing --ckpt: {args.ckpt}")
    if not args.vocab.exists():
        raise SystemExit(f"Missing --vocab: {args.vocab}")

    if args.prompt_tokens == "1,2,3":
        print("[WARN] --prompt_tokens is default placeholder (1,2,3); set it for real runs.")

    torch = __import__("torch")

    from CausalGPT import constrained_nanochat

    labels = _load_labels_from_csv(args.data_csv, delimiter=args.delimiter)

    state = torch.load(args.ckpt, map_location="cpu")
    state_dict = state.get("model", state.get("state_dict", state))

    # Prefer checkpoint-provided model_args so inference matches training.
    cfg_kwargs = None
    if isinstance(state, dict):
        cfg_kwargs = state.get("model_args")

    if isinstance(cfg_kwargs, dict) and cfg_kwargs:
        allowed = {f.name for f in dataclasses.fields(constrained_nanochat.GPTConfig)}
        filtered = {k: v for k, v in cfg_kwargs.items() if k in allowed}
        cfg = constrained_nanochat.GPTConfig(**filtered)
    else:
        cfg = constrained_nanochat.GPTConfig()
    model = constrained_nanochat.GPT(cfg)
    model.init_weights()
    model.load_state_dict(state_dict, strict=False)

    vocab = _read_vocab(args.vocab)
    if int(getattr(cfg, "vocab_size", len(vocab))) != len(vocab):
        raise ValueError(
            "Vocab size mismatch: ckpt model_args.vocab_size="
            f"{int(getattr(cfg, 'vocab_size'))} but vocab.txt has {len(vocab)} lines. "
            "Training vocab and inference vocab must be identical."
        )
    decode = _make_decode_from_vocab(vocab)
    vocab_dict = _make_vocab_dict(vocab)
    prompt_tokens = _parse_prompt_tokens(args.prompt_tokens)

    priors_anc_text = args.exdbn_anc.read_text(encoding="utf-8", errors="ignore")

    args.out_anc.parent.mkdir(parents=True, exist_ok=True)
    res = model.generate_to_anc(
        prompt_tokens,
        args.max_gen,
        decode=decode,
        out_anc_path=args.out_anc,
        node_names=labels,
        priors_anc_text=priors_anc_text,
        priors_vocab_dict=vocab_dict,
    )

    anc_path_written = res.get("anc_path")
    anc_idx_path_written = res.get("anc_idx_path")
    if anc_path_written:
        print(f"[NANOCHAT] wrote_generated_hard_anc: {anc_path_written}")
    if anc_idx_path_written:
        print(f"[NANOCHAT] wrote_generated_hard_anc_idx: {anc_idx_path_written}")

    prompt_text = res.get("prompt_text")
    raw_text = res.get("raw_text")
    if prompt_text is not None:
        print("[NANOCHAT] prompt_text:\n" + str(prompt_text))
    if raw_text is not None:
        print("[NANOCHAT] raw_text:\n" + str(raw_text))


if __name__ == "__main__":
    main()
