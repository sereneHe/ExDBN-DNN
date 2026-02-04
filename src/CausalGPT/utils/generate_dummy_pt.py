from __future__ import annotations

import sys

from CausalGPT.make_vocab_gpt4style import dummy_assets_cli
"""
生成一个 dummy .pt checkpoint 和一个对应的 vocab.txt
"""

def main() -> None:
    # Back-compat wrapper around `python -m CausalGPT.make_vocab_gpt4style dummy-assets ...`
    dummy_assets_cli(sys.argv[1:])


if __name__ == "__main__":
    main()
