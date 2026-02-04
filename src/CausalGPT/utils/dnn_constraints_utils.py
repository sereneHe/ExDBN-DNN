from __future__ import annotations

import numpy as np


def parse_anc_file(anc_path: str, n: int) -> np.ndarray:
    """Parse a CaMML-style .anc file into a forbidden-edge mask.

    We interpret lines with "i -> j" as forbidden directed edges i->j.

    Returns:
        mask: (n, n) int array with 1 meaning forbidden, 0 meaning allowed.

    Notes:
        This function is intentionally lightweight and has no torch dependency,
        so it can be used in EXDBN-only pipelines.
    """

    mask = np.zeros((n, n), dtype=int)
    with open(anc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "->" not in line:
                continue
            parts = line.strip().replace(";", "").replace("}", "").split()
            if len(parts) < 3:
                continue
            try:
                i, j = int(parts[0]), int(parts[2])
            except Exception:
                continue
            if 0 <= i < n and 0 <= j < n:
                mask[i, j] = 1
    return mask


def train_dnn_with_constraints(
    n: int,
    forbidden_mask: np.ndarray,
    *,
    target_adj: np.ndarray | None = None,
    epochs: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Lazy wrapper around the backup implementation.

    This keeps torch as an optional dependency unless DNN training is requested.
    """

    from CausalGPT.backup.dnn_constraints_utils import (  # noqa: WPS433 (runtime import)
        train_dnn_with_constraints as _impl,
    )

    return _impl(
        n,
        forbidden_mask,
        target_adj=target_adj,
        epochs=epochs,
        seed=seed,
    )
