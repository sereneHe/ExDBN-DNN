from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


def _safe_str(x: object) -> str:
    if x is None:
        return "<null>"
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ")
    return s


def _truncate(s: str, max_len: int) -> str:
    if max_len <= 0:
        return s
    return s if len(s) <= max_len else (s[: max(0, max_len - 1)] + "…")


def _iter_rows_csv_stdlib(path: Path, *, delimiter: str, max_rows: int, encoding: str) -> tuple[list[str], Iterable[list[str]]]:
    f = path.open("r", encoding=encoding, errors="replace", newline="")
    reader = csv.reader(f, delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        f.close()
        raise SystemExit(f"Empty CSV: {path}")

    def gen() -> Iterable[list[str]]:
        try:
            for i, row in enumerate(reader):
                if max_rows > 0 and i >= max_rows:
                    break
                yield row
        finally:
            f.close()

    return [str(h) for h in header], gen()


def _iter_rows_csv_pandas(path: Path, *, delimiter: str, max_rows: int) -> tuple[list[str], Iterable[list[object]]]:
    import pandas as pd

    df = pd.read_csv(path, sep=delimiter)
    if max_rows > 0:
        df = df.head(max_rows)
    header = [str(c) for c in df.columns]

    def gen() -> Iterable[list[object]]:
        for _, row in df.iterrows():
            yield row.tolist()

    return header, gen()


def _format_row(
    header: list[str],
    row: list[object],
    *,
    mode: str,
    pair_sep: str,
    kv_sep: str,
    max_cols: int,
    max_cell_chars: int,
) -> str:
    cols = header
    vals = row
    if max_cols > 0:
        cols = cols[:max_cols]
        vals = vals[:max_cols]

    if mode == "values":
        parts = [_truncate(_safe_str(v), max_cell_chars) for v in vals]
        return pair_sep.join(parts)

    if mode == "pairs":
        parts: list[str] = []
        for c, v in zip(cols, vals):
            parts.append(f"{c}{kv_sep}{_truncate(_safe_str(v), max_cell_chars)}")
        return pair_sep.join(parts)

    raise ValueError(f"Unknown mode: {mode}")


def table_to_token_text(
    in_csv: Path,
    *,
    out_txt: Path,
    delimiter: str = ",",
    mode: str = "pairs",
    include_header_line: bool = True,
    header_prefix: str = "columns:",
    max_rows: int = 1000,
    max_cols: int = 0,
    max_cell_chars: int = 64,
    pair_sep: str = " ",
    kv_sep: str = "=",
    encoding: str = "utf-8",
) -> None:
    if not in_csv.exists():
        raise SystemExit(f"Missing input CSV: {in_csv}")

    # Prefer pandas when available (handles more edge cases), fall back to stdlib.
    try:
        header, rows = _iter_rows_csv_pandas(in_csv, delimiter=delimiter, max_rows=max_rows)
    except Exception:
        header, rows = _iter_rows_csv_stdlib(in_csv, delimiter=delimiter, max_rows=max_rows, encoding=encoding)

    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with out_txt.open("w", encoding="utf-8") as f:
        if include_header_line:
            cols = header[:max_cols] if max_cols > 0 else header
            f.write(header_prefix + " " + pair_sep.join(cols) + "\n")

        for row in rows:
            # If a row is shorter than header, pad with empty.
            if len(row) < len(header):
                row = list(row) + ["" for _ in range(len(header) - len(row))]
            line = _format_row(
                header,
                list(row),
                mode=mode,
                pair_sep=pair_sep,
                kv_sep=kv_sep,
                max_cols=max_cols,
                max_cell_chars=max_cell_chars,
            )
            f.write(line + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Convert any tabular CSV into a text corpus for tokenizer training. "
            "Outputs one line per row (either 'pairs' like col=value or 'values' only)."
        )
    )
    p.add_argument("--in_csv", type=Path, required=True)
    p.add_argument("--out_txt", type=Path, required=True)
    p.add_argument("--delimiter", type=str, default=",")
    p.add_argument("--mode", type=str, choices=["pairs", "values"], default="pairs")
    p.add_argument("--include_header_line", type=int, choices=[0, 1], default=1)
    p.add_argument("--header_prefix", type=str, default="columns:")
    p.add_argument("--max_rows", type=int, default=1000, help="0 means no limit")
    p.add_argument("--max_cols", type=int, default=0, help="0 means no limit")
    p.add_argument("--max_cell_chars", type=int, default=64, help="0 means no limit")
    p.add_argument("--pair_sep", type=str, default=" ")
    p.add_argument("--kv_sep", type=str, default="=")
    p.add_argument("--encoding", type=str, default="utf-8")

    args = p.parse_args()

    table_to_token_text(
        args.in_csv,
        out_txt=args.out_txt,
        delimiter=args.delimiter,
        mode=args.mode,
        include_header_line=bool(args.include_header_line),
        header_prefix=args.header_prefix,
        max_rows=int(args.max_rows),
        max_cols=int(args.max_cols),
        max_cell_chars=int(args.max_cell_chars),
        pair_sep=args.pair_sep,
        kv_sep=args.kv_sep,
        encoding=args.encoding,
    )

    print(f"[TEXT] wrote {args.out_txt}")


if __name__ == "__main__":
    main()
