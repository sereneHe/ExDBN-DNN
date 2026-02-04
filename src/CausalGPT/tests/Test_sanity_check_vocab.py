from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _read_vocab(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty vocab file: {path}")
    return lines


def _ckpt_get_model_and_args(state: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (state_dict, model_args) from common nanoGPT/nanochat checkpoint shapes."""
    state_dict: dict[str, Any] | None = None
    model_args: dict[str, Any] | None = None

    if isinstance(state, dict):
        maybe_sd = state.get("model") or state.get("state_dict")
        if isinstance(maybe_sd, dict):
            state_dict = maybe_sd
        elif maybe_sd is not None:
            # torch can deserialize OrderedDict or other mapping types;
            # treat any mapping-like as dict for our purposes.
            try:
                state_dict = dict(maybe_sd)
            except Exception:
                state_dict = None

        maybe_args = state.get("model_args")
        if isinstance(maybe_args, dict):
            model_args = maybe_args

        if state_dict is None:
            # If the dict doesn't contain typical checkpoint metadata keys, treat it as a raw state_dict.
            reserved = {
                "model",
                "state_dict",
                "optimizer",
                "model_args",
                "iter_num",
                "best_val_loss",
            }
            if not (set(state.keys()) & reserved):
                state_dict = state

        return state_dict, model_args

    return None, None


def _extract_anc_tokens(anc_text: str) -> tuple[list[tuple[str, str, str]], set[str]]:
    """Extract arcs and node tokens from an `.anc` string.

    Accepts both `->` and `=>` arcs. Weight is required by the legacy parser
    (`CausalGPT.utils.anc_convert_token.parse_anc_arcs`), but this extractor is more
    lenient and allows missing weights.

    Returns:
        (arcs, tokens) where arcs is list of (src, op, dst).
    """
    import re

    # Examples:
    #   A1MHMS -> A1MHMU 0.99999;
    #   A1MHMS=>A3MHMS 0.00001
    #   A1MHMS -> A3MHMS;
    arc_re = re.compile(
        r"(?P<src>\w+)\s*(?P<op>->|=>)\s*(?P<dst>\w+)(?:\s+(?P<w>[-+0-9.eE]+))?\s*;?"
    )

    arcs: list[tuple[str, str, str]] = []
    tokens: set[str] = set()
    for m in arc_re.finditer(anc_text):
        src = m.group("src")
        op = m.group("op")
        dst = m.group("dst")
        arcs.append((src, op, dst))
        tokens.add(src)
        tokens.add(dst)

    return arcs, tokens


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    default_out_dir = repo_root / "reports" / "tmp_sanity"
    default_ckpt = default_out_dir / "dummy_ckpt.pt"
    default_vocab = default_out_dir / "vocab.txt"

    p = argparse.ArgumentParser(
        description=(
            "Sanity-check NanoChat artifacts compatibility: ckpt.pt + vocab.txt (+ optional priors .anc). "
            "Optionally, generate dummy ckpt+vocab via --generate 1."
        )
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=default_ckpt,
        help=(
            "Path to ckpt.pt (input for check / output for generate). "
            f"Default: {default_ckpt}"
        ),
    )
    p.add_argument(
        "--vocab",
        type=Path,
        default=default_vocab,
        help=(
            "Path to vocab.txt (input for check / output for generate). "
            f"Default: {default_vocab}"
        ),
    )
    p.add_argument(
        "--generate",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "1=generate a dummy ckpt (.pt) AND vocab.txt using CausalGPT.make_vocab_gpt4style "
            "(overwrites --ckpt/--vocab). 0=run sanity check (default)."
        ),
    )
    p.add_argument(
        "--vocab_size",
        type=int,
        default=64,
        help="When --generate 1: number of vocab lines + model vocab_size (default: 64)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="When --generate 1: optional CSV whose header columns will be included in vocab.txt",
    )
    p.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="CSV delimiter for --csv (default: ,)",
    )
    p.add_argument(
        "--also_add_basic_unk_pad_bos_eos",
        type=int,
        choices=[0, 1],
        default=1,
        help="When --generate 1: also add <unk>,<pad>,<bos>,<eos> at the top (default: 1)",
    )
    p.add_argument("--sequence_len", type=int, default=128, help="When --generate 1: model sequence length")
    p.add_argument("--n_layer", type=int, default=2, help="When --generate 1")
    p.add_argument("--n_head", type=int, default=2, help="When --generate 1")
    p.add_argument("--n_kv_head", type=int, default=2, help="When --generate 1")
    p.add_argument("--n_embd", type=int, default=64, help="When --generate 1")
    p.add_argument(
        "--anc",
        type=Path,
        action="append",
        default=[],
        help="Optional .anc priors to validate against vocab tokens (can repeat)",
    )
    p.add_argument(
        "--strict",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=error on missing anc tokens; 0=warn only (default: 1)",
    )
    p.add_argument("--max_missing_show", type=int, default=50, help="Max missing tokens to print")

    args = p.parse_args()

    if args.generate == 1:
        from CausalGPT.make_vocab_gpt4style import write_dummy_nanochat_assets, write_vocab_txt

        # 1) Write a dummy ckpt with matching vocab_size.
        # 2) Overwrite vocab.txt using write_vocab_txt so it contains desired tokens.
        args.ckpt.parent.mkdir(parents=True, exist_ok=True)
        args.vocab.parent.mkdir(parents=True, exist_ok=True)

        write_dummy_nanochat_assets(
            out_ckpt=args.ckpt,
            out_vocab=args.vocab,
            vocab_size=int(args.vocab_size),
            sequence_len=int(args.sequence_len),
            n_layer=int(args.n_layer),
            n_head=int(args.n_head),
            n_kv_head=int(args.n_kv_head),
            n_embd=int(args.n_embd),
        )

        write_vocab_txt(
            out_vocab=args.vocab,
            vocab_size=int(args.vocab_size),
            csv_path=args.csv,
            delimiter=str(args.delimiter),
            also_add_basic_unk_pad_bos_eos=bool(args.also_add_basic_unk_pad_bos_eos),
        )

        print(f"[OK] generated ckpt:  {args.ckpt}")
        print(f"[OK] generated vocab: {args.vocab}")
        return

    if not args.ckpt.exists():
        raise SystemExit(
            f"Missing --ckpt: {args.ckpt}\n"
            "Tip: run with --generate 1 to create dummy assets at the default paths."
        )
    if not args.vocab.exists():
        raise SystemExit(
            f"Missing --vocab: {args.vocab}\n"
            "Tip: run with --generate 1 to create dummy assets at the default paths."
        )
    for ap in args.anc:
        if not ap.exists():
            raise SystemExit(f"Missing --anc: {ap}")

    torch = __import__("torch")

    vocab = _read_vocab(args.vocab)
    vocab_len = len(vocab)
    vocab_dict = {tok: i for i, tok in enumerate(vocab)}

    state = torch.load(args.ckpt, map_location="cpu")
    state_dict, model_args = _ckpt_get_model_and_args(state)

    if state_dict is None:
        raise SystemExit(
            "Checkpoint does not look like a supported nanoGPT/nanochat checkpoint. "
            "Expected a dict with a 'model' (state_dict) key, or a raw state_dict mapping."
        )

    ckpt_vocab_size: int | None = None
    if model_args is None:
        print("[WARN] ckpt has no 'model_args' dict; cannot validate vocab_size against training config")
    else:
        vocab_size = model_args.get("vocab_size")
        if vocab_size is None:
            print("[WARN] ckpt model_args has no 'vocab_size'; cannot validate vocab size")
        else:
            try:
                vocab_size_int = int(vocab_size)
            except Exception:
                raise SystemExit(f"Invalid ckpt model_args.vocab_size (not int): {vocab_size!r}")

            ckpt_vocab_size = vocab_size_int

            if vocab_size_int != vocab_len:
                raise SystemExit(
                    "Vocab size mismatch: ckpt model_args.vocab_size="
                    f"{vocab_size_int} but vocab.txt has {vocab_len} lines. "
                    "Training vocab and inference vocab must be identical."
                )

    if vocab[:4] != ["<unk>", "<pad>", "<bos>", "<eos>"]:
        print(
            "[WARN] vocab.txt does not start with '<unk>,<pad>,<bos>,<eos>' in the first 4 lines. "
            "This may be fine if your training vocab differs, but ensure training/inference use the same file."
        )

    any_fail = False
    anc_validated = 0
    for anc_path in args.anc:
        anc_text = anc_path.read_text(encoding="utf-8", errors="ignore")
        arcs, tokens = _extract_anc_tokens(anc_text)

        if "arcs{" not in anc_text:
            print(f"[WARN] {anc_path}: missing 'arcs{{' header (still attempting to parse arcs)")

        if not arcs:
            print(f"[WARN] {anc_path}: no arcs parsed")
            continue

        anc_validated += 1

        missing = sorted(t for t in tokens if t not in vocab_dict)
        if missing:
            msg = (
                f"{anc_path}: {len(missing)}/{len(tokens)} unique node tokens are missing from vocab.txt. "
                "This will cause KeyError inside parse_anc_arcs() and priors won't apply."
            )
            if args.strict == 1:
                print("[ERROR] " + msg)
                any_fail = True
            else:
                print("[WARN] " + msg)

            shown = missing[: max(0, int(args.max_missing_show))]
            if shown:
                print("[MISSING_TOKENS] " + ", ".join(shown))
            if len(missing) > len(shown):
                print(f"[MISSING_TOKENS] ... and {len(missing) - len(shown)} more")
        else:
            print(f"[OK] {anc_path}: all {len(tokens)} unique node tokens found in vocab")

    if any_fail:
        raise SystemExit(2)

    # Basic success summary
    print("[OK] sanity check passed")
    print(f"[SUMMARY] ckpt={args.ckpt}")
    print(f"[SUMMARY] vocab={args.vocab} (lines={vocab_len})")
    if ckpt_vocab_size is not None:
        print(f"[SUMMARY] ckpt.model_args.vocab_size={ckpt_vocab_size}")
    if args.anc:
        print(f"[SUMMARY] anc_files={len(args.anc)} validated={anc_validated}")
    else:
        print("[SUMMARY] anc_files=0")


if __name__ == "__main__":
    main()
