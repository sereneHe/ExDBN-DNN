from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pytest


def test_priors_affect_generation_deterministic() -> None:
    torch = pytest.importorskip("torch")

    from CausalGPT.constrained_nanochat import GPT, GPTConfig

    class DummyGPT(GPT):
        def __init__(self, cfg: GPTConfig, *, preferred: int) -> None:
            super().__init__(cfg)
            self._preferred = int(preferred)

        def forward(self, idx):  # type: ignore[override]
            b, t = idx.shape
            v = int(self.config.vocab_size)
            logits = torch.full((b, t, v), -10.0, dtype=torch.float32, device=idx.device)
            logits[:, :, self._preferred] = 10.0
            return logits

    cfg = GPTConfig(sequence_len=8, vocab_size=8, n_layer=1, n_head=1, n_kv_head=1, n_embd=16)
    model = DummyGPT(cfg, preferred=0)
    model.init_weights()

    vocab_dict = {"tok0": 0, "tok1": 1, "tok2": 2}
    anc_text = """
arcs{
  tok0 -> tok0 0.9;
  tok0 -> tok1 0.9;
    tok0 => tok2 0.99999;
}
"""

    out_no_anc = list(model.generate([3], max_tokens=5, temperature=0.0))
    out_with_anc = list(
        model.generate(
        [3],
        max_tokens=5,
        temperature=0.0,
        anc_text=anc_text,
        vocab_dict=vocab_dict,
        )
    )

    assert all(int(t) == 0 for t in out_no_anc)
    assert 1 not in [int(t) for t in out_with_anc], "banned token should never appear"
    assert 2 in [int(t) for t in out_with_anc], "replace prior should bias toward token2"


def _load_labels_from_csv(path: Path, *, delimiter: str = ",") -> list[str] | None:
    try:
        import pandas as pd

        df0 = pd.read_csv(path, sep=delimiter, nrows=0)
        cols = [str(c) for c in df0.columns]
        return cols if cols else None
    except Exception:
        return None


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


def _parse_tabu_edges_from_anc_idx(anc_idx_path: Path, *, n_nodes: int) -> list[tuple[int, int]]:
    from CausalGPT.utils.dnn_constraints_utils import parse_anc_file

    forbidden = parse_anc_file(str(anc_idx_path), n_nodes)
    tabu: list[tuple[int, int]] = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if int(forbidden[i, j]) == 1:
                tabu.append((i, j))
    return tabu


def _parse_tabu_edges_from_named_anc(
    anc_path: Path,
    *,
    label_to_idx: dict[str, int],
) -> list[tuple[int, int]]:
    """Parse tabu edges from a named CaMML-style .anc file.

    Only hard constraints with "A -> B" are turned into tabu edges.
    Soft constraints with "A => B" are ignored for EXDBN tabu_edges.
    """

    tabu: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for line in anc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "->" not in line:
            continue
        parts = line.strip().replace(";", "").replace("}", "").split()
        if len(parts) < 3:
            continue
        a, b = parts[0], parts[2]
        if a not in label_to_idx or b not in label_to_idx:
            continue
        i, j = int(label_to_idx[a]), int(label_to_idx[b])
        if i == j:
            continue
        e = (i, j)
        if e not in seen:
            seen.add(e)
            tabu.append(e)
    return tabu


def _shd(a: np.ndarray, b: np.ndarray) -> int:
    a = (a != 0).astype(int)
    b = (b != 0).astype(int)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = (a != b).astype(int)
    np.fill_diagonal(diff, 0)
    return int(diff.sum())


def test_e2e_nanochat_anc_exdbn_compare(tmp_path: Path) -> None:
    # EXDBN runs Gurobi/MILP and can be very memory-hungry. Make this test opt-in.
    # Run with: RUN_E2E_EXDBN=1 python -m pytest -q src/CausalGPT/tests/test.py -s
    if os.getenv("SKIP_E2E_EXDBN") == "1":
        pytest.skip("SKIP_E2E_EXDBN=1")
    if os.getenv("RUN_E2E_EXDBN") != "1":
        pytest.skip("Set RUN_E2E_EXDBN=1 to run EXDBN e2e (Gurobi/MILP).")

    torch = pytest.importorskip("torch")

    from CausalGPT import constrained_nanochat, exdbn_ban_edges

    data_path = Path(os.getenv("E2E_DATA_PATH", "/Users/xiaoyuhe/Datasets/CoDiet/codiet_302_0.csv"))
    if not data_path.exists():
        pytest.skip(f"Dataset not found: {data_path}")

    # Conservative defaults to avoid OOM by accident.
    degree = int(os.getenv("E2E_MAX_DEGREE", "2"))
    skiprows = int(os.getenv("E2E_SKIPROWS", "1"))
    delimiter = os.getenv("E2E_DELIMITER", ",")
    sample_size_env = os.getenv("E2E_SAMPLE_SIZE", "50")
    if sample_size_env is None:
        sample_size = 50
    else:
        s = str(sample_size_env).strip().lower()
        if s in {"", "none", "null"}:
            sample_size = None
        else:
            v = int(s)
            sample_size = None if v <= 0 else v

    labels = _load_labels_from_csv(data_path, delimiter=delimiter)
    if labels is None:
        pytest.skip("CSV header labels required for NanoChat anc_idx generation.")
    label_to_idx = {str(name): int(i) for i, name in enumerate(labels)}

    # 1) Baseline EXDBN
    adj_base, info_base = exdbn_ban_edges.predict_exdbn_adjacency_from_csv(
        data_path,
        sample_size=sample_size,
        max_degree=degree,
        skiprows=skiprows,
        delimiter=delimiter,
        tabu_edges=None,
    )
    n_nodes = int(adj_base.shape[0])

    # 2) Build constraints (existing constraints + optional NanoChat-generated constraints)
    constrained_anc_idx_env = os.getenv("E2E_CONSTRAINED_ANC_IDX")
    constrained_anc_idx: Path | None = Path(constrained_anc_idx_env) if constrained_anc_idx_env else None

    constrained_anc_env = os.getenv("E2E_CONSTRAINED_ANC")
    constrained_anc: Path | None = Path(constrained_anc_env) if constrained_anc_env else None

    # NanoChat generation is best-effort by default. Set SKIP_NANOCHAT=1 to disable.
    ckpt = os.getenv("E2E_NANOCHAT_CKPT")
    vocab_path = os.getenv("E2E_NANOCHAT_VOCAB")
    prompt_tokens_str = os.getenv("E2E_NANOCHAT_PROMPT_TOKENS")
    max_gen = int(os.getenv("E2E_NANOCHAT_MAX_GEN", "512"))
    do_nanochat = os.getenv("SKIP_NANOCHAT") != "1"

    if do_nanochat and not (ckpt and vocab_path and prompt_tokens_str):
        # Fill placeholder defaults (per user request) and warn.
        ckpt = ckpt or "/path/to/ckpt.pt"
        vocab_path = vocab_path or "/path/to/vocab.txt"
        prompt_tokens_str = prompt_tokens_str or "1,2,3"
        print(
            "[WARN] NanoChat enabled but E2E_NANOCHAT_* env vars missing; using placeholder defaults: "
            f"E2E_NANOCHAT_CKPT={ckpt} E2E_NANOCHAT_VOCAB={vocab_path} E2E_NANOCHAT_PROMPT_TOKENS={prompt_tokens_str}"
        )

    # Only run NanoChat if ckpt/vocab actually exist.
    if do_nanochat:
        try:
            ckpt_ok = bool(ckpt) and Path(ckpt).exists()
            vocab_ok = bool(vocab_path) and Path(vocab_path).exists()
        except Exception:
            ckpt_ok = False
            vocab_ok = False
        if not (ckpt_ok and vocab_ok):
            print(
                "[WARN] NanoChat disabled because ckpt/vocab path does not exist. "
                "Provide real E2E_NANOCHAT_CKPT and E2E_NANOCHAT_VOCAB to enable."
            )
            do_nanochat = False

    if (constrained_anc_idx is None and constrained_anc is None) and (not do_nanochat):
        raise RuntimeError(
            "No constraints source available. Provide E2E_CONSTRAINED_ANC_IDX or E2E_CONSTRAINED_ANC, "
            "or provide NanoChat env vars (E2E_NANOCHAT_CKPT/E2E_NANOCHAT_VOCAB/E2E_NANOCHAT_PROMPT_TOKENS)."
        )

    tabu_edges_set: set[tuple[int, int]] = set()
    if constrained_anc_idx is not None:
        assert constrained_anc_idx.exists(), f"Missing anc_idx file: {constrained_anc_idx}"
        tabu_edges_set.update(_parse_tabu_edges_from_anc_idx(constrained_anc_idx, n_nodes=n_nodes))
    if constrained_anc is not None:
        assert constrained_anc.exists(), f"Missing anc file: {constrained_anc}"
        tabu_edges_set.update(_parse_tabu_edges_from_named_anc(constrained_anc, label_to_idx=label_to_idx))

    if do_nanochat:
        state = torch.load(Path(ckpt), map_location="cpu")
        state_dict = state.get("model", state.get("state_dict", state))

        cfg = constrained_nanochat.GPTConfig()
        model = constrained_nanochat.GPT(cfg)
        model.init_weights()
        model.load_state_dict(state_dict, strict=False)

        vocab = _read_vocab(Path(vocab_path))
        decode = _make_decode_from_vocab(vocab)
        vocab_dict = _make_vocab_dict(vocab)
        prompt_tokens = [int(x) for x in prompt_tokens_str.split(",") if x.strip()]

        priors_anc_text = None
        if constrained_anc is not None:
            priors_anc_text = constrained_anc.read_text(encoding="utf-8", errors="ignore")

        out_anc_env = os.getenv("E2E_NANOCHAT_OUT_ANC")
        if out_anc_env:
            out_anc_path = Path(out_anc_env)
        else:
            repo_root = Path(__file__).resolve().parents[3]
            out_dir = repo_root / "reports" / "causalgpt_runs_codiet" / "codiet"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_anc_path = out_dir / f"nanochat_generated_hard_{data_path.stem}.anc"
        res = model.generate_to_anc(
            prompt_tokens,
            max_gen,
            decode=decode,
            out_anc_path=out_anc_path,
            node_names=labels,
            priors_anc_text=priors_anc_text,
            priors_vocab_dict=vocab_dict if priors_anc_text is not None else None,
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

        anc_idx_path = res.get("anc_idx_path")
        if anc_idx_path:
            tabu_edges_set.update(_parse_tabu_edges_from_anc_idx(Path(anc_idx_path), n_nodes=n_nodes))

    tabu_edges = sorted(tabu_edges_set)
    print(f"[E2E] tabu_edges={len(tabu_edges)}")

    # 3) Constrained EXDBN
    adj_con, info_con = exdbn_ban_edges.predict_exdbn_adjacency_from_csv(
        data_path,
        sample_size=sample_size,
        max_degree=degree,
        skiprows=skiprows,
        delimiter=delimiter,
        tabu_edges=tabu_edges,
    )
    assert adj_con.shape == adj_base.shape

    # 4) Verify hard constraints are respected
    for i, j in tabu_edges:
        assert int(adj_con[i, j]) == 0

    # 5) Compare
    shd = _shd(adj_base, adj_con)
    print(
        "[E2E] done | "
        f"base_nnz={(adj_base!=0).sum()} con_nnz={(adj_con!=0).sum()} shd={shd} "
        f"base_gap={info_base.get('gap')} con_gap={info_con.get('gap')}"
    )
