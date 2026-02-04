from __future__ import annotations

import argparse
from pathlib import Path
import time
import re

import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from CausalGPT import exdbn_ban_edges
"""ExDBN + DNN + 约束推断主脚本。"""


def _try_save_adj_heatmap(
    adj: np.ndarray,
    *,
    labels: list[str] | None,
    out_path: Path,
    title: str,
) -> bool:
    """Best-effort adjacency heatmap writer.

    Uses matplotlib if available; otherwise returns False without failing the run.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    n = int(adj.shape[0])
    if n <= 30:
        figsize = (6, 6)
        fontsize = 7
    elif n <= 60:
        figsize = (9, 9)
        fontsize = 6
    else:
        figsize = (10, 10)
        fontsize = 0

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(adj, cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_title(title)

    if labels is not None and n <= 60:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(labels, fontsize=fontsize)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def _try_save_network_plot(
    adj: np.ndarray,
    *,
    labels: list[str] | None,
    out_path: Path,
    title: str,
    seed: int = 42,
) -> bool:
    """Best-effort directed network plot writer.

    Uses networkx + matplotlib if available; otherwise returns False.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import networkx as nx
    except Exception:
        return False

    adj_bin = (adj != 0).astype(int)
    n = int(adj_bin.shape[0])
    node_labels = labels if labels is not None else [str(i) for i in range(n)]

    g = nx.DiGraph()
    g.add_nodes_from(node_labels)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj_bin[i, j] != 0:
                g.add_edge(node_labels[i], node_labels[j])

    pos = nx.spring_layout(g, seed=seed)

    if n <= 30:
        figsize = (8, 8)
        node_size = 900
        font_size = 7
        width = 1.2
    elif n <= 60:
        figsize = (10, 10)
        node_size = 450
        font_size = 5
        width = 0.8
    else:
        figsize = (12, 12)
        node_size = 120
        font_size = 0
        width = 0.5

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.axis("off")

    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        width=width,
        connectionstyle="arc3,rad=0.08",
    )
    if font_size > 0:
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=font_size)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True

def _sniff_delimiter(path: Path, *, default: str = ",") -> str:
    """Detect whether a CSV is comma- or semicolon-delimited.

    Uses the first non-empty line (typically the header) and compares counts.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                comma_count = line.count(",")
                semicolon_count = line.count(";")
                if semicolon_count > comma_count:
                    return ";"
                if comma_count > semicolon_count:
                    return ","
                return default
    except OSError:
        return default
    return default


def _compute_metrics(
    learned_adj: np.ndarray,
    *,
    true_adj: np.ndarray | None = None,
    gap: float | None = None,
) -> dict:
    """Compute metrics for a learned adjacency.

    - Always reports: gap (if provided) and nnz.
    - If a true adjacency is available and dagsolvers is importable, reports
      fdr/tpr/fpr/shd/precision/f1score/g_score using dagsolvers.metrics_utils.count_accuracy.
    - Otherwise leaves those fields as NaN (so downstream aggregation can ignore).
    """

    learned_bin = (learned_adj != 0).astype(int)

    metrics = {
        "gap": float(gap) if gap is not None else np.nan,
        "fdr": np.nan,
        "tpr": np.nan,
        "fpr": np.nan,
        "shd": np.nan,
        "nnz": int(learned_bin.sum()),
        "precision": np.nan,
        "f1score": np.nan,
        "g_score": np.nan,
    }

    if true_adj is None:
        return metrics

    try:
        import importlib

        metrics_utils = importlib.import_module("dagsolvers.metrics_utils")
        count_accuracy = getattr(metrics_utils, "count_accuracy")
    except Exception:
        return metrics

    true_bin = (true_adj != 0).astype(int)
    try:
        metrics_full = count_accuracy(true_bin, learned_bin, [], [], test_dag=True)
    except Exception:
        return metrics

    for k in ("fdr", "tpr", "fpr", "shd", "precision", "f1score", "g_score", "nnz"):
        if k in metrics_full:
            metrics[k] = metrics_full.get(k)
    # Ensure nnz is consistent even if missing in metrics_full
    metrics["nnz"] = int(learned_bin.sum())
    return metrics


def ensure_minimal_anc_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text("arcs{\n}\n")


def _sanitize_feature_labels(raw_labels: list[object]) -> list[str]:
    """Sanitize feature labels for .anc output.

    CaMML-style .anc is token-based; whitespace and some symbols can break parsing.
    This keeps the original names readable while making them safe.
    """

    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for lab in raw_labels:
        s = str(lab).strip()
        s = re.sub(r"\s+", "_", s)
        s = s.replace(";", "_").replace(",", "_")
        s = s.replace("=>", "_").replace("->", "_")
        if not s:
            s = "col"
        count = seen.get(s, 0)
        seen[s] = count + 1
        cleaned.append(s if count == 0 else f"{s}_{count}")
    return cleaned


def _normalize_codiet_column_name(name: object) -> str:
    s = str(name)
    # CoDiet headers often contain numeric-looking category suffixes like ".0".
    # Normalizing early keeps downstream ANC tokens compatible with vocab that uses integer forms.
    if s.endswith(".0"):
        s = s[: -len(".0")]
    return s


def _normalize_codiet_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Normalize CoDiet-style column names (e.g., VST_1.0 -> VST_1) with de-dup."""

    raw_cols = [str(c) for c in df.columns]
    norm_cols = [_normalize_codiet_column_name(c) for c in raw_cols]

    # De-duplicate while preserving order (avoid accidental collisions after normalization).
    seen: dict[str, int] = {}
    out_cols: list[str] = []
    for c in norm_cols:
        k = c
        count = seen.get(k, 0)
        seen[k] = count + 1
        out_cols.append(k if count == 0 else f"{k}_{count}")

    changed = out_cols != raw_cols
    if changed:
        df = df.copy()
        df.columns = out_cols
    return df, changed


def _assert_no_dot_zero_headers(columns: list[object], *, context: str) -> None:
    bad = [str(c) for c in columns if str(c).endswith(".0")]
    if bad:
        preview = bad[:10]
        more = "" if len(bad) <= 10 else f" (+{len(bad) - 10} more)"
        raise ValueError(f"Found headers ending with '.0' in {context}: {preview}{more}")


def _project_root() -> Path:
    # Prefer the CausalGPT subproject root if present (so YAML lives under src/CausalGPT/configs).
    # Fallback to the monorepo root layout.
    p = Path(__file__).resolve()
    causalgpt_root = p.parents[1]
    if (causalgpt_root / "pyproject.toml").exists():
        return causalgpt_root
    # .../ExDBN-DNN/src/CausalGPT/tests/ExDBN_perform.py -> .../ExDBN-DNN
    return p.parents[3]


def _default_config_path() -> Path:
    return _project_root() / "configs" / "exdbn_perform.yaml"


def _load_cfg(path: Path):
    if path is None or not path.exists():
        return OmegaConf.create({})
    return OmegaConf.load(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="YAML config path (OmegaConf). CLI args override config.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to a single CSV file, or directory containing batch CSVs."
    )
    parser.add_argument(
        "--batch_glob",
        type=str,
        default=None,
        help="Glob pattern for batch CSVs (e.g. 'codiet_302_*.csv'). If set, runs batch mode."
    )
    parser.add_argument(
        "--anc_path",
        type=Path,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="If set, only use the first K columns/features of the input CSV. Useful for quick smoke tests on high-dimensional datasets.",
    )
    parser.add_argument(
        "--save_generated_csv_to_data_dir",
        action="store_true",
        help=(
            "If set and the dataset is CoDiet (codiet*), save the generated input CSV (after header normalization and optional max_features cut) "
            "back into the same directory as the original CSV so you can inspect it."
        ),
    )
    parser.add_argument(
        "--check_no_dot_zero_headers",
        action="store_true",
        help="If set, fail if any resulting column name ends with '.0' (e.g., VST_1.0).",
    )
    parser.add_argument(
        "--only_normalize_and_save",
        action="store_true",
        help="If set, only read/normalize/save (and optionally check) the CSV, then exit without running EXDBN/DNN.",
    )
    parser.add_argument(
        "--skip_dnn",
        action="store_true",
        help="Skip the DNN+constraints stage (useful if torch is unavailable). Also implied when --epochs <= 0.",
    )
    parser.add_argument(
        "--require_dnn",
        action="store_true",
        help="Fail if the DNN stage does not run (useful to verify torch+DNN are actually enabled).",
    )
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Prefix for constraint outputs (writes <prefix>_hard_constraints.txt, <prefix>_prior_constraints.txt, <prefix>.anc). Default: <out_dir>/ExDBN_LLM",
    )
    parser.add_argument("--write_anc", action="store_true", help="Also write CaMML-style .anc to out_prefix.anc")
    parser.add_argument(
        "--plot_adj",
        action="store_true",
        help="If set, also save adjacency heatmaps (.png) next to the saved adjacency matrices.",
    )
    parser.add_argument(
        "--plot_network",
        action="store_true",
        help="If set, also save directed network plots (.net.png) next to the saved adjacency matrices.",
    )
    parser.add_argument("--conf", type=float, default=0.99999)
    parser.add_argument("--ancs", type=str, default="[]", help="Soft ancestor priors as python literal list, e.g. '[(0,2)]'")
    parser.add_argument("--forb_ancs", type=str, default="[]", help="Soft forbidden ancestor priors as python literal list")
    parser.add_argument(
        "--align_asia",
        action="store_true",
        help="If set, reorder asia_*.csv columns to match the Asia mapping order (prevents header/order mismatch).",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    defaults = OmegaConf.create(
        {
            "data_path": "/Users/xiaoyuhe/Datasets/Asia/asia_250_0.csv",
            "anc_path": str(_project_root() / "reports" / "test_ancs.anc"),
            "out_dir": str(_project_root() / "reports"),
        }
    )
    cfg = OmegaConf.merge(defaults, cfg)

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Batch mode: process all files matching batch_glob
    batch_files = []
    if args.batch_glob:
        import glob
        batch_files = sorted(glob.glob(str(args.batch_glob)))
    elif args.data_path and args.data_path.is_dir():
        batch_files = sorted([str(f) for f in args.data_path.glob("*.csv")])

    if batch_files:
        print(f"Batch mode: found {len(batch_files)} files.")
        for batch_file in batch_files:
            print(f"Processing batch file: {batch_file}")
            # Set up output prefix for each batch
            batch_stem = Path(batch_file).stem
            out_prefix = str(out_dir / batch_stem / "ExDBN_LLM")
            # ...existing code...
            # 1. 读取数据
            delimiter = _sniff_delimiter(Path(batch_file))
            df_data = pd.read_csv(batch_file, sep=delimiter)

            is_codiet = Path(batch_file).stem.lower().startswith("codiet")
            codiet_changed = False
            if is_codiet:
                df_data, codiet_changed = _normalize_codiet_columns(df_data)

            if args.max_features is not None:
                if args.max_features <= 0:
                    raise ValueError("--max_features must be > 0")
                if df_data.shape[1] > args.max_features:
                    df_data = df_data.iloc[:, : args.max_features]
            X = df_data.to_numpy()
            n, d = X.shape

            if args.check_no_dot_zero_headers:
                _assert_no_dot_zero_headers(list(df_data.columns), context=f"{batch_file} (post-normalization)")

            if is_codiet and args.save_generated_csv_to_data_dir and Path(batch_file).suffix.lower() == ".csv":
                out_in_data_dir: Path
                if args.max_features is not None:
                    out_in_data_dir = Path(batch_file).parent / f"__tmp_{batch_stem}_first{d}_normalized.csv"
                else:
                    out_in_data_dir = Path(batch_file).parent / f"{batch_stem}_normalized.csv"
                df_data.to_csv(out_in_data_dir, index=False, sep=",")
                print(f"[CODIET] wrote generated CSV: {out_in_data_dir}")

            if args.only_normalize_and_save:
                continue

            # 2. EXDBN 推断
            data_path_for_exdbn = batch_file
            delimiter_for_exdbn = delimiter

            # If we changed CoDiet column names and we're not writing the cropped tmp CSV,
            # persist a normalized copy and feed it to EXDBN.
            if is_codiet and codiet_changed and args.max_features is None and Path(batch_file).suffix.lower() == ".csv":
                norm_csv = out_dir / batch_stem / f"__tmp_{batch_stem}_normalized.csv"
                df_data.to_csv(norm_csv, index=False, sep=",")
                data_path_for_exdbn = norm_csv
                delimiter_for_exdbn = ","

            if args.max_features is not None and Path(batch_file).suffix.lower() == ".csv":
                tmp_csv = out_dir / batch_stem / f"__tmp_{batch_stem}_first{d}.csv"
                df_data.to_csv(tmp_csv, index=False, sep=",")
                data_path_for_exdbn = tmp_csv
                delimiter_for_exdbn = ","
            start = time.time()
            try:
                adj_exdbn, _info = exdbn_ban_edges.predict_exdbn_adjacency_from_csv(
                    data_path_for_exdbn,
                    sample_size=None,
                    max_degree=args.degree,
                    delimiter=delimiter_for_exdbn,
                    skiprows=1,
                )
            except Exception as e:
                raise RuntimeError(
                    f"EXDBN prediction failed for {batch_file}. Ensure dagsolvers + solver deps are installed. "
                    f"(original: {type(e).__name__}: {e})"
                ) from e
            runtime_exdbn = time.time() - start
            # ...existing code for constraint writing, DNN, etc. (copy from single file mode)...
            exdbn_gap = None
            try:
                exdbn_gap = _info.get("gap")
            except Exception:
                exdbn_gap = None
            ancs = eval(args.ancs)
            forb_ancs = eval(args.forb_ancs)
            feature_labels = _sanitize_feature_labels(list(df_data.columns))
            write_anc = bool(args.write_anc) or ((not args.skip_dnn) and (args.epochs is not None) and (args.epochs > 0))
            _complement, banned_edges = exdbn_ban_edges.ban_edges(
                adj_exdbn,
                max_power=4,
                out_prefix=out_prefix,
                write_anc=write_anc,
                conf=args.conf,
                anc=ancs,
                forb_anc=forb_ancs,
                labels=feature_labels,
            )
            exdbn_anc_path = Path(out_prefix).with_suffix(".anc")
            exdbn_anc_idx_path = Path(f"{out_prefix}_idx.anc")
            if write_anc:
                exdbn_ban_edges.write_camml_anc(
                    exdbn_anc_idx_path,
                    n_nodes=d,
                    conf=args.conf,
                    anc=ancs,
                    forb_anc=forb_ancs,
                    abs_edges=banned_edges,
                    labels=None,
                )
            # DNN stage (optional)
            do_dnn = (not args.skip_dnn) and (args.epochs is not None) and (args.epochs > 0)
            if args.require_dnn and not do_dnn:
                raise RuntimeError(
                    "DNN stage is required (--require_dnn) but is disabled by flags. "
                    "Remove --skip_dnn and set --epochs > 0."
                )
            if do_dnn:
                try:
                    from CausalGPT.utils.dnn_constraints_utils import parse_anc_file, train_dnn_with_constraints
                except Exception as e:
                    raise RuntimeError(
                        "DNN stage requested but failed to import dependencies (likely PyTorch). "
                        "If you want to force DNN, use a Python environment where torch is installable (often Python 3.10/3.11). "
                        "Otherwise rerun with --skip_dnn or set EPOCHS=0 / pass --epochs 0."
                    ) from e
                # Prefer feeding DNN with constraints derived from EXDBN prediction.
                # ...existing DNN code...
        print("Batch processing complete.")
        return

    # Single file mode (original logic)
    data_path = Path(args.data_path) if args.data_path is not None else Path(cfg.data_path)
    anc_path = Path(args.anc_path) if args.anc_path is not None else Path(cfg.anc_path)
    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = str(out_dir / "ExDBN_LLM")
    delimiter = _sniff_delimiter(data_path)
    df_data = pd.read_csv(data_path, sep=delimiter)

    is_codiet = data_path.stem.lower().startswith("codiet")
    codiet_changed = False
    if is_codiet:
        df_data, codiet_changed = _normalize_codiet_columns(df_data)

    if args.max_features is not None:
        if args.max_features <= 0:
            raise ValueError("--max_features must be > 0")
        if df_data.shape[1] > args.max_features:
            df_data = df_data.iloc[:, : args.max_features]
    X = df_data.to_numpy()
    n, d = X.shape

    if args.check_no_dot_zero_headers:
        _assert_no_dot_zero_headers(list(df_data.columns), context=f"{data_path} (post-normalization)")

    if is_codiet and args.save_generated_csv_to_data_dir and data_path.suffix.lower() == ".csv":
        out_in_data_dir: Path
        if args.max_features is not None:
            out_in_data_dir = data_path.parent / f"__tmp_{data_path.stem}_first{d}_normalized.csv"
        else:
            out_in_data_dir = data_path.parent / f"{data_path.stem}_normalized.csv"
        df_data.to_csv(out_in_data_dir, index=False, sep=",")
        print(f"[CODIET] wrote generated CSV: {out_in_data_dir}")

    if args.only_normalize_and_save:
        print("[INFO] --only_normalize_and_save set; exiting before EXDBN.")
        return

    data_path_for_exdbn = data_path
    delimiter_for_exdbn = delimiter

    if is_codiet and codiet_changed and args.max_features is None and data_path.suffix.lower() == ".csv":
        norm_csv = out_dir / f"__tmp_{data_path.stem}_normalized.csv"
        df_data.to_csv(norm_csv, index=False, sep=",")
        data_path_for_exdbn = norm_csv
        delimiter_for_exdbn = ","

    if args.max_features is not None and data_path.suffix.lower() == ".csv":
        tmp_csv = out_dir / f"__tmp_{data_path.stem}_first{d}.csv"
        df_data.to_csv(tmp_csv, index=False, sep=",")
        data_path_for_exdbn = tmp_csv
        delimiter_for_exdbn = ","
    start = time.time()
    try:
        adj_exdbn, _info = exdbn_ban_edges.predict_exdbn_adjacency_from_csv(
            data_path_for_exdbn,
            sample_size=None,
            max_degree=args.degree,
            delimiter=delimiter_for_exdbn,
            skiprows=1,
        )
    except Exception as e:
        raise RuntimeError(
            "EXDBN prediction failed. Ensure dagsolvers + solver deps are installed, or run in the bestdagsolver env. "
            f"(original: {type(e).__name__}: {e})"
        ) from e
    runtime_exdbn = time.time() - start

    exdbn_gap = None
    try:
        exdbn_gap = _info.get("gap")
    except Exception:
        exdbn_gap = None

    # 2b. Generate constraints from EXDBN-predicted adjacency
    ancs = eval(args.ancs)
    forb_ancs = eval(args.forb_ancs)

    feature_labels = _sanitize_feature_labels(list(df_data.columns))

    # If we plan to run DNN, ensure we also write the EXDBN-derived .anc file so
    # the DNN can consume the same constraints.
    write_anc = bool(args.write_anc) or ((not args.skip_dnn) and (args.epochs is not None) and (args.epochs > 0))
    _complement, banned_edges = exdbn_ban_edges.ban_edges(
        adj_exdbn,
        max_power=4,
        out_prefix=out_prefix,
        write_anc=write_anc,
        conf=args.conf,
        anc=ancs,
        forb_anc=forb_ancs,
        labels=feature_labels,
    )

    exdbn_anc_path = Path(out_prefix).with_suffix(".anc")
    exdbn_anc_idx_path = Path(f"{out_prefix}_idx.anc")

    # Also write an index-based .anc for tooling that expects integer node IDs (e.g. the DNN parser).
    if write_anc:
        exdbn_ban_edges.write_camml_anc(
            exdbn_anc_idx_path,
            n_nodes=d,
            conf=args.conf,
            anc=ancs,
            forb_anc=forb_ancs,
            abs_edges=banned_edges,
            labels=None,
        )

    # 3. DNN + constraints 推断 (optional)
    do_dnn = (not args.skip_dnn) and (args.epochs is not None) and (args.epochs > 0)
    if args.require_dnn and not do_dnn:
        raise RuntimeError(
            "DNN stage is required (--require_dnn) but is disabled by flags. "
            "Remove --skip_dnn and set --epochs > 0."
        )
    if do_dnn:
        try:
            from CausalGPT.utils.dnn_constraints_utils import parse_anc_file, train_dnn_with_constraints
        except Exception as e:  # torch import can fail on unsupported Python versions
            raise RuntimeError(
                "DNN stage requested but failed to import dependencies (likely PyTorch). "
                "If you want to force DNN, use a Python environment where torch is installable (often Python 3.10/3.11). "
                "Otherwise rerun with --skip_dnn or set EPOCHS=0 / pass --epochs 0."
            ) from e

        # Prefer feeding DNN with constraints derived from EXDBN prediction.
        anc_path_for_dnn = exdbn_anc_idx_path if exdbn_anc_idx_path.exists() else (exdbn_anc_path if exdbn_anc_path.exists() else anc_path)
        ensure_minimal_anc_file(anc_path_for_dnn)
        mask = parse_anc_file(str(anc_path_for_dnn), d)
        start = time.time()
        adj_dnn = train_dnn_with_constraints(
            d,
            mask,
            target_adj=adj_exdbn,
            epochs=args.epochs,
            seed=args.seed,
        )
        runtime_dnn = time.time() - start
        print(f"[INFO] Ran DNN stage: epochs={args.epochs}, runtime_seconds={runtime_dnn:.3f}")
    else:
        # If DNN is skipped, still treat the "DNN output" as the EXDBN prediction
        # (user intent: put EXDBN prediction into DNN).
        adj_dnn = (adj_exdbn != 0).astype(int)
        runtime_dnn = 0.0

    # 3b. Persist learned adjacency matrices (useful when no ground truth exists)
    adj_exdbn_bin = (adj_exdbn != 0).astype(int)
    adj_dnn_bin = (adj_dnn != 0).astype(int)

    adj_exdbn_csv = out_dir / f"adj_exdbn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.csv"
    adj_dnn_csv = out_dir / f"adj_dnn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.csv"
    pd.DataFrame(adj_exdbn_bin, index=feature_labels, columns=feature_labels).to_csv(adj_exdbn_csv)
    pd.DataFrame(adj_dnn_bin, index=feature_labels, columns=feature_labels).to_csv(adj_dnn_csv)

    adj_exdbn_npy = out_dir / f"adj_exdbn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.npy"
    adj_dnn_npy = out_dir / f"adj_dnn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.npy"
    np.save(adj_exdbn_npy, adj_exdbn_bin)
    np.save(adj_dnn_npy, adj_dnn_bin)

    adj_exdbn_png = out_dir / f"adj_exdbn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.png"
    adj_dnn_png = out_dir / f"adj_dnn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.png"
    if args.plot_adj:
        ok1 = _try_save_adj_heatmap(
            adj_exdbn_bin,
            labels=feature_labels,
            out_path=adj_exdbn_png,
            title=adj_exdbn_csv.name,
        )
        ok2 = _try_save_adj_heatmap(
            adj_dnn_bin,
            labels=feature_labels,
            out_path=adj_dnn_png,
            title=adj_dnn_csv.name,
        )
        if not (ok1 and ok2):
            print("[INFO] --plot_adj requested but matplotlib is unavailable; skipped PNG plots.")

    adj_exdbn_net_png = out_dir / f"adj_exdbn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.net.png"
    adj_dnn_net_png = out_dir / f"adj_dnn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.net.png"
    if args.plot_network:
        ok1 = _try_save_network_plot(
            adj_exdbn_bin,
            labels=feature_labels,
            out_path=adj_exdbn_net_png,
            title=adj_exdbn_csv.name,
        )
        ok2 = _try_save_network_plot(
            adj_dnn_bin,
            labels=feature_labels,
            out_path=adj_dnn_net_png,
            title=adj_dnn_csv.name,
        )
        if not (ok1 and ok2):
            print("[INFO] --plot_network requested but networkx/matplotlib is unavailable; skipped network plots.")

    # 4. 评估
    # NOTE: for real datasets (e.g. CoDiet) we typically do not have ground truth,
    # so metrics like SHD/FDR will be NaN. For synthetic datasets with known truth,
    # pass true_adj in the future.
    metrics_exdbn = _compute_metrics(adj_exdbn, true_adj=None, gap=exdbn_gap)
    metrics_dnn = _compute_metrics(adj_dnn, true_adj=None, gap=np.nan)

    # 5. 保存结果
    columns = [
        "dataset",
        "features",
        "samples",
        "degree",
        "runtime_seconds",
        "gap",
        "fdr",
        "tpr",
        "fpr",
        "shd",
        "nnz",
        "precision",
        "f1score",
        "g_score",
    ]

    row_exdbn = dict(
        dataset=data_path.stem,
        features=d,
        samples=n,
        degree=args.degree,
        runtime_seconds=runtime_exdbn,
        **metrics_exdbn,
    )
    row_dnn = dict(
        dataset=data_path.stem,
        features=d,
        samples=n,
        degree=args.degree,
        runtime_seconds=runtime_dnn,
        **metrics_dnn,
    )

    exdbn_metrics_csv = out_dir / f"exdbn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.csv"
    dnn_metrics_csv = out_dir / f"exdbn_dnn_{data_path.stem}_d{d}_n{n}_deg{args.degree}.csv"
    (pd.DataFrame([row_exdbn])[columns]).to_csv(exdbn_metrics_csv, index=False)
    (pd.DataFrame([row_dnn])[columns]).to_csv(dnn_metrics_csv, index=False)

    print(f"[OUT] exdbn_metrics: {exdbn_metrics_csv}")
    print(f"[OUT] dnn_metrics:  {dnn_metrics_csv}")
    print(f"[OUT] adj_exdbn:    {adj_exdbn_csv}")
    print(f"[OUT] adj_dnn:     {adj_dnn_csv}")
    if args.plot_adj and adj_exdbn_png.exists():
        print(f"[OUT] adj_exdbn_png:{adj_exdbn_png}")
    if args.plot_adj and adj_dnn_png.exists():
        print(f"[OUT] adj_dnn_png: {adj_dnn_png}")
    if args.plot_network and adj_exdbn_net_png.exists():
        print(f"[OUT] adj_exdbn_net:{adj_exdbn_net_png}")
    if args.plot_network and adj_dnn_net_png.exists():
        print(f"[OUT] adj_dnn_net: {adj_dnn_net_png}")
    if write_anc and exdbn_anc_path.exists():
        print(f"[OUT] anc_named:   {exdbn_anc_path}")
    if write_anc and exdbn_anc_idx_path.exists():
        print(f"[OUT] anc_idx:     {exdbn_anc_idx_path}")
    if not do_dnn:
        print("[INFO] Skipped DNN stage (--skip_dnn or --epochs <= 0).")
    print("结果已保存。")


if __name__ == "__main__":
    main()
