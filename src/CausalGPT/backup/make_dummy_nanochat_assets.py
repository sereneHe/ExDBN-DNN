from __future__ import annotations

import argparse
from pathlib import Path

from CausalGPT.make_vocab_gpt4style import write_dummy_nanochat_assets


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Create a dummy (randomly initialized) NanoChat-style checkpoint and vocab.txt for pipeline smoke tests. "
            "This is NOT a trained model; it only validates the code path."
        )
    )
    p.add_argument("--out_ckpt", type=Path, required=True, help="Output .pt checkpoint path")
    p.add_argument("--out_vocab", type=Path, required=True, help="Output vocab.txt path")
    p.add_argument("--vocab_size", type=int, default=64, help="Vocab size (default: 64)")
    p.add_argument("--sequence_len", type=int, default=128, help="Sequence length (default: 128)")
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_kv_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=64)
    args = p.parse_args()

    write_dummy_nanochat_assets(
        out_ckpt=args.out_ckpt,
        out_vocab=args.out_vocab,
        vocab_size=int(args.vocab_size),
        sequence_len=int(args.sequence_len),
        n_layer=int(args.n_layer),
        n_head=int(args.n_head),
        n_kv_head=int(args.n_kv_head),
        n_embd=int(args.n_embd),
    )


if __name__ == "__main__":
    main()
