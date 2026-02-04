from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def _load_labels_from_csv(path: Path, *, delimiter: str = ",") -> list[str] | None:
    try:
        df0 = pd.read_csv(path, sep=delimiter, nrows=0)
        cols = [str(c) for c in df0.columns]
        return cols if cols else None
    except Exception:
        return None


def _read_vocab(path: Path) -> list[str]:
    # One token-piece per line (common for simple tokenizers).
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty vocab file: {path}")
    return lines


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


def _shd(a: np.ndarray, b: np.ndarray) -> int:
    a = (a != 0).astype(int)
    b = (b != 0).astype(int)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = (a != b).astype(int)
    np.fill_diagonal(diff, 0)
    return int(diff.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare EXDBN vs EXDBN+constraints(from .anc_idx or constrained-nanochat generation).")

    ap.add_argument("--data_path", type=str, required=True, help="CSV or NPZ dataset path")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")

    ap.add_argument("--sample_size", type=int, default=None, help="Rows used for EXDBN (CSV: optional; NPZ: required)")
    ap.add_argument("--degree", type=int, default=5, help="EXDBN max in/out degree")
    ap.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    ap.add_argument("--skiprows", type=int, default=1, help="CSV skiprows passed to np.loadtxt")

    ap.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Short dataset name used in output filenames (default: 'codiet' for codiet_* inputs, else data_path.stem)",
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step calls and key artifacts.",
    )

    # Baseline constraints (derived from baseline EXDBN adjacency) are optional artifacts.
    ap.add_argument("--write_baseline_anc", action="store_true", help="Also write baseline constraints .anc/.anc_idx derived from baseline adjacency")
    ap.add_argument("--max_power", type=int, default=4, help="Reachability power for baseline ban_edges()")
    ap.add_argument("--conf", type=float, default=0.99999, help="Constraint confidence (anc weight = 1-conf)")

    # Constrained run inputs
    ap.add_argument("--constrained_anc_idx", type=str, default=None, help="Path to an _idx.anc file to use as hard constraints (tabu edges)")

    # Optional: generate constrained anc via constrained_nanochat (requires ckpt + vocab + prompt tokens)
    ap.add_argument("--nanochat_ckpt", type=str, default=None, help="Optional: torch checkpoint/state_dict for constrained_nanochat.GPT")
    ap.add_argument("--nanochat_vocab", type=str, default=None, help="Optional: vocab file (one token piece per line) for decoding generated ids")
    ap.add_argument("--nanochat_prompt_tokens", type=str, default=None, help="Optional: comma-separated prompt token ids")
    ap.add_argument("--nanochat_max_gen", type=int, default=512, help="Optional: max new tokens to generate")
    ap.add_argument("--nanochat_out_anc", type=str, default=None, help="Optional: output .anc path for nanochat generation")

    args = ap.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_name is not None and args.dataset_name.strip() != "":
        dataset_name = args.dataset_name.strip()
    else:
        stem = data_path.stem
        dataset_name = "codiet" if stem.lower().startswith("codiet_") else stem

    # Ensure local package imports work
    import sys

    # .../src is the import root containing the CausalGPT package
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from CausalGPT import exdbn_ban_edges

    labels = None
    if data_path.suffix.lower() not in {".npz"}:
        labels = _load_labels_from_csv(data_path, delimiter=args.delimiter)

    if args.verbose:
        print("[CALL] exdbn_ban_edges.predict_exdbn_adjacency_from_{csv|npz}(tabu_edges=None)")
        print(f"[ARG] data_path={data_path}")
        print(f"[ARG] sample_size={args.sample_size} degree={args.degree}")
        if labels is not None:
            print(f"[ARG] labels_from_header={len(labels)}")

    # 1) Baseline EXDBN
    if data_path.suffix.lower() == ".npz":
        if args.sample_size is None:
            raise SystemExit("For .npz input, --sample_size is required")
        adj_exdbn, info_base = exdbn_ban_edges.predict_exdbn_adjacency_from_npz(
            data_path,
            sample_size=args.sample_size,
            max_degree=args.degree,
            tabu_edges=None,
        )
    else:
        adj_exdbn, info_base = exdbn_ban_edges.predict_exdbn_adjacency_from_csv(
            data_path,
            sample_size=args.sample_size,
            max_degree=args.degree,
            skiprows=args.skiprows,
            delimiter=args.delimiter,
            tabu_edges=None,
        )

    n_nodes = int(adj_exdbn.shape[0])

    if args.verbose:
        print(f"[RET] baseline_adj shape={adj_exdbn.shape} nnz={(adj_exdbn!=0).sum()}")

    # Save baseline adjacency
    base_npy = out_dir / f"adj_{dataset_name}_exdbn.npy"
    base_csv = out_dir / f"adj_{dataset_name}_exdbn.csv"
    np.save(base_npy, adj_exdbn)
    pd.DataFrame(adj_exdbn).to_csv(base_csv, index=False)

    # Optional baseline anc artifacts
    if args.write_baseline_anc:
        prefix = out_dir / "baseline"
        exdbn_ban_edges.ban_edges(
            (adj_exdbn != 0).astype(int),
            max_power=args.max_power,
            out_prefix=str(prefix),
            write_anc=True,
            conf=args.conf,
            labels=labels,
        )
        # Always write an idx version too
        exdbn_ban_edges.ban_edges(
            (adj_exdbn != 0).astype(int),
            max_power=args.max_power,
            out_prefix=str(prefix) + "_idx",
            write_anc=True,
            conf=args.conf,
            labels=[str(i) for i in range(n_nodes)],
        )

    constrained_anc_idx = Path(args.constrained_anc_idx) if args.constrained_anc_idx else None

    # 2) Optional: generate constrained anc via constrained_nanochat
    if constrained_anc_idx is None and args.nanochat_ckpt and args.nanochat_vocab and args.nanochat_prompt_tokens and args.nanochat_out_anc:
        import torch

        from CausalGPT import constrained_nanochat

        ckpt_path = Path(args.nanochat_ckpt)
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("model", state.get("state_dict", state))

        cfg = constrained_nanochat.GPTConfig()
        model = constrained_nanochat.GPT(cfg)
        model.init_weights()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {len(unexpected)}")

        vocab = _read_vocab(Path(args.nanochat_vocab))
        decode = _make_decode_from_vocab(vocab)
        prompt_tokens = [int(x) for x in args.nanochat_prompt_tokens.split(",") if x.strip()]

        out_anc_path = Path(args.nanochat_out_anc)
        res = model.generate_to_anc(
            prompt_tokens,
            args.nanochat_max_gen,
            decode=decode,
            out_anc_path=out_anc_path,
            node_names=labels,
        )

        if args.verbose:
            prompt_text = res.get("prompt_text")
            raw_text = res.get("raw_text")
            if prompt_text is not None:
                print("[NANOCHAT] prompt_text:")
                print(prompt_text)
            if raw_text is not None:
                print("[NANOCHAT] raw_text:")
                print(raw_text)
        if res.get("anc_idx_path"):
            constrained_anc_idx = Path(res["anc_idx_path"])
        else:
            # If no labels, we may only have a named anc; for EXDBN constraints we need idx.
            raise SystemExit("Nanochat generation succeeded but did not produce anc_idx (need --labels / CSV header)")

    if constrained_anc_idx is None:
        raise SystemExit("No constrained constraints provided. Pass --constrained_anc_idx, or provide --nanochat_* generation args.")

    if args.verbose:
        print("[CALL] _parse_tabu_edges_from_anc_idx -> parse_anc_file")
        print(f"[ARG] constrained_anc_idx={constrained_anc_idx}")

    # 3) Rerun EXDBN with tabu edges parsed from constrained anc
    tabu_edges = _parse_tabu_edges_from_anc_idx(constrained_anc_idx, n_nodes=n_nodes)

    if args.verbose:
        print(f"[RET] tabu_edges={len(tabu_edges)}")
        print("[CALL] exdbn_ban_edges.predict_exdbn_adjacency_from_{csv|npz}(tabu_edges=tabu_edges)")

    if data_path.suffix.lower() == ".npz":
        adj_con, info_con = exdbn_ban_edges.predict_exdbn_adjacency_from_npz(
            data_path,
            sample_size=args.sample_size,
            max_degree=args.degree,
            tabu_edges=tabu_edges,
        )
    else:
        adj_con, info_con = exdbn_ban_edges.predict_exdbn_adjacency_from_csv(
            data_path,
            sample_size=args.sample_size,
            max_degree=args.degree,
            skiprows=args.skiprows,
            delimiter=args.delimiter,
            tabu_edges=tabu_edges,
        )

    con_npy = out_dir / f"adj_{dataset_name}_exdbn_constrained_NanoChat.npy"
    con_csv = out_dir / f"adj_{dataset_name}_exdbn_constrained_NanoChat.csv"
    np.save(con_npy, adj_con)
    pd.DataFrame(adj_con).to_csv(con_csv, index=False)

    if args.verbose:
        print(f"[RET] constrained_adj shape={adj_con.shape} nnz={(adj_con!=0).sum()}")

    # 4) Diffs + metrics
    base_bin = (adj_exdbn != 0).astype(int)
    con_bin = (adj_con != 0).astype(int)
    np.fill_diagonal(base_bin, 0)
    np.fill_diagonal(con_bin, 0)

    added = np.argwhere((base_bin == 0) & (con_bin == 1))
    removed = np.argwhere((base_bin == 1) & (con_bin == 0))

    def _edge_rows(arr: np.ndarray, change: str) -> list[dict]:
        rows: list[dict] = []
        for i, j in arr.tolist():
            rows.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "src": labels[i] if labels else str(i),
                    "dst": labels[j] if labels else str(j),
                    "change": change,
                }
            )
        return rows

    diff_rows = _edge_rows(added, "added") + _edge_rows(removed, "removed")
    pd.DataFrame(diff_rows).to_csv(out_dir / "edge_diffs.csv", index=False)

    summary = {
        "data_path": str(data_path),
        "n_nodes": n_nodes,
        "baseline": {
            **info_base,
            "nnz": int(base_bin.sum()),
        },
        "constrained": {
            **info_con,
            "nnz": int(con_bin.sum()),
            "tabu_edges": int(len(tabu_edges)),
            "constrained_anc_idx": str(constrained_anc_idx),
        },
        "diff": {
            "shd": _shd(base_bin, con_bin),
            "added": int(len(added)),
            "removed": int(len(removed)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.verbose:
        print(f"[METRIC] shd={summary['diff']['shd']} added={summary['diff']['added']} removed={summary['diff']['removed']}")

    print("[OUT]", base_csv)
    print("[OUT]", con_csv)
    print("[OUT]", out_dir / "edge_diffs.csv")
    print("[OUT]", out_dir / "summary.json")


if __name__ == "__main__":
    main()
