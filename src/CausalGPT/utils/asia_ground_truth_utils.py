from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AsiaGroundTruth:
    variable_order: list[str]
    adjacency: np.ndarray


def load_asia_ground_truth(
    *,
    mapping_path: Path,
    graph_path: Path,
) -> AsiaGroundTruth:
    variable_order = [
        line.strip()
        for line in Path(mapping_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]

    adj = np.loadtxt(str(graph_path), dtype=int)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Ground truth adjacency must be square, got {adj.shape}")
    if len(variable_order) != adj.shape[0]:
        raise ValueError(
            f"Mapping length {len(variable_order)} != adjacency size {adj.shape[0]}"
        )

    return AsiaGroundTruth(variable_order=variable_order, adjacency=adj)


def reorder_asia_csv_to_mapping(
    *,
    csv_path: Path,
    out_csv_path: Path | None,
    mapping_path: Path,
    header_aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Reorder Asia CSV columns to match mapping order.

    The mapping file uses long names (e.g., visit_to_asia) while common datasets
    use short headers (e.g., asia). Provide header_aliases to map short->long.

    Returns (df_reordered, info).
    """
    header_aliases = header_aliases or {
        "asia": "visit_to_asia",
        "tub": "tuberculosis",
        "smoke": "smoking",
        "lung": "lung_cancer",
        "bronc": "bronchitis",
        "either": "either_turb_or_lung_cancer",
        "xray": "positive_xray",
        "dysp": "dyspnoea",
    }

    df = pd.read_csv(csv_path)
    original_cols = list(df.columns)

    # Normalize columns to mapping names
    normalized_cols: list[str] = []
    for c in original_cols:
        normalized_cols.append(header_aliases.get(c, c))

    mapping_order = [
        line.strip()
        for line in Path(mapping_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]

    # Build a view with normalized names but keep original data
    df_norm = df.copy()
    df_norm.columns = normalized_cols

    missing = [name for name in mapping_order if name not in df_norm.columns]
    extra = [name for name in df_norm.columns if name not in mapping_order]
    if missing:
        raise ValueError(f"CSV missing required columns from mapping: {missing}")

    df_out = df_norm[mapping_order].copy()

    if out_csv_path is not None:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv_path, index=False)

    info = {
        "csv_path": str(csv_path),
        "out_csv_path": str(out_csv_path) if out_csv_path is not None else None,
        "original_columns": original_cols,
        "normalized_columns": normalized_cols,
        "mapping_order": mapping_order,
        "extra_columns_ignored": extra,
    }
    return df_out, info
