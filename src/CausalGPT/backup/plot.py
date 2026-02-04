from __future__ import annotations

"""Plot EXDBN/DNN adjacency matrices.

This script unifies:
- heatmap plotting (formerly plot_adjacency.py)
- directed network plotting (formerly plot_network.py)

Inputs:
- one or more adjacency files: adj_exdbn_*.csv/.npy, adj_dnn_*.csv/.npy
- and/or directories containing those files

Outputs (by default next to each input file):
- heatmap: *.png
- network: *.net.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _default_out_dir() -> Path:
    # .../ExDBN-DNN/src/CausalGPT/plot.py -> .../ExDBN-DNN/reports/figures
    return Path(__file__).resolve().parents[2] / "reports" / "figures"


def _load_adj(path: Path) -> tuple[np.ndarray, list[str] | None]:
    if path.suffix.lower() == ".npy":
        return np.load(path), None

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, index_col=0)
        labels = [str(x) for x in df.index.tolist()]
        return df.to_numpy(), labels

    raise ValueError(f"Unsupported adjacency format: {path} (expected .csv or .npy)")


def _candidate_adj_files(dir_path: Path) -> list[Path]:
    files: list[Path] = []
    for pat in ("adj_exdbn_*.csv", "adj_dnn_*.csv", "adj_exdbn_*.npy", "adj_dnn_*.npy"):
        files.extend(sorted(dir_path.glob(pat)))

    # If both CSV and NPY exist for the same stem, keep CSV.
    by_stem: dict[str, Path] = {}
    for p in files:
        key = p.with_suffix("").name
        existing = by_stem.get(key)
        if existing is None:
            by_stem[key] = p
            continue
        if existing.suffix.lower() == ".npy" and p.suffix.lower() == ".csv":
            by_stem[key] = p

    return sorted(by_stem.values())


def _save_heatmap(
    adj: np.ndarray,
    *,
    labels: list[str] | None,
    out_path: Path,
    title: str | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    if title:
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


def _save_network_png(
    adj: np.ndarray,
    *,
    labels: list[str] | None,
    out_path: Path,
    title: str,
    seed: int,
    layout: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import networkx as nx

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

    if layout == "circular":
        pos = nx.circular_layout(g)
    else:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot EXDBN/DNN adjacency matrices as heatmaps and/or networks.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Adjacency files (.csv/.npy) or directories containing adj_exdbn_*/adj_dnn_* outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=["heatmap", "network", "both"],
        default="both",
        help="Which plots to generate.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=_default_out_dir(),
        help="Directory to write PNGs into. Default: ExDBN-DNN/reports/figures (override with --out_dir).",
    )
    parser.add_argument("--layout", choices=["spring", "circular"], default="spring", help="Network layout.")
    parser.add_argument("--seed", type=int, default=42, help="Network layout RNG seed.")
    args = parser.parse_args()

    targets: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            targets.extend(_candidate_adj_files(p))
        else:
            targets.append(p)

    if not targets:
        raise SystemExit("No adjacency files found.")

    for path in targets:
        adj, labels = _load_adj(path)
        title = path.name

        base_name = path.with_suffix("").name

        args.out_dir.mkdir(parents=True, exist_ok=True)
        heatmap_png = args.out_dir / (base_name + ".png")
        net_png = args.out_dir / (base_name + ".net.png")

        if args.mode in ("heatmap", "both"):
            _save_heatmap(adj, labels=labels, out_path=heatmap_png, title=title)
            print(f"[HEATMAP] {path} -> {heatmap_png}")

        if args.mode in ("network", "both"):
            _save_network_png(
                adj,
                labels=labels,
                out_path=net_png,
                title=title,
                seed=args.seed,
                layout=args.layout,
            )
            print(f"[NETWORK] {path} -> {net_png}")


if __name__ == "__main__":
    main()
