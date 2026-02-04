from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _assert_no_dot_zero_headers(columns: list[object], *, context: str) -> None:
    bad = [str(c) for c in columns if str(c).endswith(".0")]
    if bad:
        preview = bad[:10]
        more = "" if len(bad) <= 10 else f" (+{len(bad) - 10} more)"
        raise ValueError(f"Found headers ending with '.0' in {context}: {preview}{more}")


def generate_replicates(
    *,
    input_csv: Path,
    out_dir: Path,
    n_reps: int,
    seed: int,
    n_rows: int | None,
    overwrite: bool,
) -> list[Path]:
    """Generate bootstrap replicates codiet_<N>_<r>.csv from a base CoDiet CSV."""

    df = pd.read_csv(input_csv)
    if df.shape[0] <= 0 or df.shape[1] <= 0:
        raise ValueError(f"Empty dataset: {input_csv}")

    _assert_no_dot_zero_headers(list(df.columns), context=str(input_csv))

    out_dir.mkdir(parents=True, exist_ok=True)

    base_n = int(df.shape[0])
    n_out = int(n_rows) if n_rows is not None else base_n
    if n_out <= 0:
        raise ValueError("n_rows must be > 0")

    out_paths: list[Path] = []
    for r in range(int(n_reps)):
        rng = np.random.default_rng(int(seed) + r)
        idx = rng.choice(base_n, size=n_out, replace=True)
        df_rep = df.iloc[idx].reset_index(drop=True)

        out_path = out_dir / f"codiet_{n_out}_{r}.csv"
        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path} (exists; pass --overwrite to replace)")
            continue

        df_rep.to_csv(out_path, index=False)

        _assert_no_dot_zero_headers(list(df_rep.columns), context=str(out_path))
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate CoDiet bootstrap replicates codiet_<N>_<r>.csv.")
    ap.add_argument(
        "--input_csv",
        type=Path,
        default=Path("/Users/xiaoyuhe/Datasets/CoDiet/codiet.csv"),
        help="Base CoDiet CSV (must already have no headers ending with .0).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/Users/xiaoyuhe/Datasets/CoDiet"),
        help="Output directory for codiet_<N>_<r>.csv files.",
    )
    ap.add_argument("--n_reps", type=int, default=6, help="Number of replicates to generate.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed. Replicate r uses seed+ r.")
    ap.add_argument(
        "--n_rows",
        type=int,
        default=None,
        help="Rows per replicate (default: same as input rows).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing codiet_<N>_<r>.csv if present.")

    args = ap.parse_args()
    outs = generate_replicates(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        n_reps=args.n_reps,
        seed=args.seed,
        n_rows=args.n_rows,
        overwrite=bool(args.overwrite),
    )

    for p in outs:
        print("[OUT]", p)


if __name__ == "__main__":
    main()
