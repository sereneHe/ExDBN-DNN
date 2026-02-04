"""GPT-4-style tokenization utilities.

This module serves two related purposes:

1) A lightweight `vocab.txt` generator used by the EXDBN→NanoChat pipeline.
    It only needs standard Python + (optionally) pandas.

2) (Optional) GPT-4 style BPE tokenizer training/inference helpers. These require
    extra dependencies (`tokenizers`, `rustbpe`, `tiktoken`) and are gated behind
    optional imports.
"""

from __future__ import annotations

import argparse
import copy
import os
from functools import lru_cache
from pathlib import Path
import sys

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def _read_csv_header_tokens(csv_path: Path, *, delimiter: str = ",") -> list[str]:
    """Read CSV header tokens (column names)."""

    try:
        import pandas as pd

        df0 = pd.read_csv(csv_path, sep=delimiter, nrows=0)
        return [str(c) for c in df0.columns]
    except Exception:
        # Pandas may not be installed in some environments. Fall back to stdlib.
        import csv

        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                raise SystemExit(f"Empty CSV: {csv_path}")
        return [str(c) for c in header]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def write_vocab_txt(
    *,
    out_vocab: Path,
    vocab_size: int,
    csv_path: Path | None = None,
    delimiter: str = ",",
    also_add_basic_unk_pad_bos_eos: bool = True,
) -> None:
    """Write a `vocab.txt` file.

    The file is a newline-separated list of tokens.

    Notes:
      - This is NOT BPE training; it is a convenience vocab generator.
      - For pipeline compatibility, keep the vocab identical between training/inference.
    """

    if vocab_size < 16:
        raise SystemExit("--vocab_size must be >= 16")

    tokens: list[str] = []
    if also_add_basic_unk_pad_bos_eos:
        tokens.extend(["<unk>", "<pad>", "<bos>", "<eos>"])
    tokens.extend(SPECIAL_TOKENS)

    if csv_path is not None:
        if not csv_path.exists():
            raise SystemExit(f"Missing --csv: {csv_path}")
        tokens.extend(_read_csv_header_tokens(csv_path, delimiter=delimiter))

    tokens = _dedupe_preserve_order(tokens)

    # Pad with deterministic filler tokens.
    i = 0
    token_set = set(tokens)
    while len(tokens) < vocab_size:
        t = f"tok{i}"
        if t not in token_set:
            tokens.append(t)
            token_set.add(t)
        i += 1

    # Truncate if oversized.
    tokens = tokens[:vocab_size]

    out_vocab.parent.mkdir(parents=True, exist_ok=True)
    out_vocab.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    print(f"[VOCAB] wrote {out_vocab} (size={len(tokens)})")


def write_dummy_nanochat_assets(
    *,
    out_ckpt: Path,
    out_vocab: Path,
    vocab_size: int = 64,
    sequence_len: int = 128,
    n_layer: int = 2,
    n_head: int = 2,
    n_kv_head: int = 2,
    n_embd: int = 64,
) -> None:
    """Create a dummy (randomly initialized) NanoChat-style ckpt + vocab.txt.

    This is NOT a trained model; it only validates the code path.
    The dummy vocab is intentionally simple: `<unk>,<pad>,<bos>,<eos>` followed by `tok*`.
    """

    if vocab_size < 8:
        raise SystemExit("--vocab_size must be >= 8")

    try:
        import torch
    except Exception as e:
        raise SystemExit(f"torch is required to create dummy assets: {e}")

    from collections import OrderedDict

    from CausalGPT import constrained_nanochat

    cfg = constrained_nanochat.GPTConfig(
        sequence_len=int(sequence_len),
        vocab_size=int(vocab_size),
        n_layer=int(n_layer),
        n_head=int(n_head),
        n_kv_head=int(n_kv_head),
        n_embd=int(n_embd),
    )
    model = constrained_nanochat.GPT(cfg)
    model.init_weights()

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_vocab.parent.mkdir(parents=True, exist_ok=True)

    model_args = {
        "sequence_len": int(cfg.sequence_len),
        "vocab_size": int(cfg.vocab_size),
        "n_layer": int(cfg.n_layer),
        "n_head": int(cfg.n_head),
        "n_kv_head": int(cfg.n_kv_head),
        "n_embd": int(cfg.n_embd),
        "window_pattern": str(getattr(cfg, "window_pattern", "SSSL")),
    }
    ckpt = {
        "model": OrderedDict(model.state_dict()),
        "optimizer": {},
        "model_args": model_args,
        "iter_num": 0,
        "best_val_loss": float("inf"),
    }
    torch.save(ckpt, out_ckpt)

    special = ["<unk>", "<pad>", "<bos>", "<eos>"]
    remaining = int(vocab_size) - len(special)
    vocab = special + [f"tok{i}" for i in range(remaining)]
    out_vocab.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    print(f"[DUMMY] wrote ckpt:  {out_ckpt}")
    print(f"[DUMMY] wrote vocab: {out_vocab} (size={len(vocab)})")

_HAS_HF_TOKENIZERS = False
try:
    # Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, pre_tokenizers, Regex
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    _HAS_HF_TOKENIZERS = True
except ModuleNotFoundError:
    HFTokenizer = None
    pre_tokenizers = decoders = Regex = None
    BPE = BpeTrainer = None

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        if not _HAS_HF_TOKENIZERS:
            raise ModuleNotFoundError("Missing optional dependency: tokenizers")
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        if not _HAS_HF_TOKENIZERS:
            raise ModuleNotFoundError("Missing optional dependency: tokenizers")
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        if not _HAS_HF_TOKENIZERS:
            raise ModuleNotFoundError("Missing optional dependency: tokenizers")
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

_HAS_RUSTBPE_TIKTOKEN = False
try:
    # Tokenizer based on rustbpe + tiktoken combo
    import pickle
    import rustbpe
    import tiktoken

    _HAS_RUSTBPE_TIKTOKEN = True
except ModuleNotFoundError:
    pickle = None
    rustbpe = None
    tiktoken = None

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        if not _HAS_RUSTBPE_TIKTOKEN:
            raise ModuleNotFoundError("Missing optional dependencies: rustbpe and/or tiktoken")
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        if not _HAS_RUSTBPE_TIKTOKEN:
            raise ModuleNotFoundError("Missing optional dependencies: rustbpe and/or tiktoken")
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        if not _HAS_RUSTBPE_TIKTOKEN:
            raise ModuleNotFoundError("Missing optional dependencies: rustbpe and/or tiktoken")
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    try:
        from nanochat.common import get_base_dir
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nanochat is not installed; get_tokenizer() is unavailable in this environment"
        ) from e
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    try:
        from nanochat.common import get_base_dir
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nanochat is not installed; get_token_bytes() is unavailable in this environment"
        ) from e
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes


def _main_vocab(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="make_vocab_gpt4style.py",
        description=(
            "Generate a newline-separated vocab.txt (special tokens + optional CSV header tokens), "
            "without requiring tokenizer dependencies."
        ),
    )
    p.add_argument("--out_vocab", type=Path, required=True, help="Output vocab.txt path")
    p.add_argument("--vocab_size", type=int, required=True, help="Number of tokens (lines) to write")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV to add header tokens (column names)")
    p.add_argument("--delimiter", type=str, default=",", help="CSV delimiter (default: ,)")
    p.add_argument(
        "--also_add_basic_unk_pad_bos_eos",
        type=int,
        choices=[0, 1],
        default=1,
        help="Also add <unk>,<pad>,<bos>,<eos> at the top (default: 1)",
    )
    args = p.parse_args(argv)

    write_vocab_txt(
        out_vocab=args.out_vocab,
        vocab_size=int(args.vocab_size),
        csv_path=args.csv,
        delimiter=str(args.delimiter),
        also_add_basic_unk_pad_bos_eos=bool(args.also_add_basic_unk_pad_bos_eos),
    )


def vocab_cli(argv: list[str]) -> None:
    """CLI entry for vocab.txt generation.

    This is a stable wrapper so other modules can delegate without duplicating argparse.
    """

    _main_vocab(argv)


def _main_dummy_assets(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="make_vocab_gpt4style.py dummy-assets",
        description=(
            "Create a dummy (randomly initialized) NanoChat-style checkpoint and vocab.txt for pipeline smoke tests. "
            "This is NOT a trained model; it only validates the code path."
        ),
    )
    p.add_argument("--out_ckpt", type=Path, required=True, help="Output .pt checkpoint path")
    p.add_argument("--out_vocab", type=Path, required=True, help="Output vocab.txt path")
    p.add_argument("--vocab_size", type=int, default=64, help="Vocab size (default: 64)")
    p.add_argument("--sequence_len", type=int, default=128, help="Sequence length (default: 128)")
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_kv_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=64)
    args = p.parse_args(argv)

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


def dummy_assets_cli(argv: list[str]) -> None:
    """CLI entry for dummy NanoChat ckpt+vocab generation.

    This is a stable wrapper so other modules can delegate without duplicating argparse.
    """

    _main_dummy_assets(argv)


def main() -> None:
    # Backward compatible behavior:
    #   python -m CausalGPT.make_vocab_gpt4style --out_vocab ... --vocab_size ...
    # New behavior:
    #   python -m CausalGPT.make_vocab_gpt4style dummy-assets --out_ckpt ... --out_vocab ...
    argv = sys.argv[1:]
    if argv[:1] == ["dummy-assets"]:
        _main_dummy_assets(argv[1:])
        return
    if argv[:1] == ["vocab"]:
        _main_vocab(argv[1:])
        return
    _main_vocab(argv)


if __name__ == "__main__":
    main()