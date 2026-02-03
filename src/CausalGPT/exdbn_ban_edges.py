from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

"""
ExDBN 边禁止工具集。
提供生成 CaMML-style 约束文件（.anc）和评估约束满足度的功能。
"""

# ---------------------------------------------------------------------
# Repo path
# ---------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_dagsolvers():
    """Import dagsolvers lazily.

    This file is used both as an EXDBN runner and as a small constraints utility.
    Tests for constraint helpers should not require dagsolvers to be installed.
    """
    def _try_import():
        from dagsolvers.data_generation_loading_utils import load_problem_dict
        from dagsolvers.dagsolver_utils import ExDagDataException
        from dagsolvers.metrics_utils import count_accuracy
        from dagsolvers.solve_milp import solve as solve_milp
        return load_problem_dict, ExDagDataException, count_accuracy, solve_milp

    try:
        return _try_import()
    except ModuleNotFoundError:
        # Try local checkout paths commonly used in this workspace.
        candidates: list[Path] = []
        env_root = os.getenv("EXDBN_DAGSOLVERS_ROOT")
        if env_root:
            candidates.append(Path(env_root))
        candidates.extend(
            [
                Path("/Users/xiaoyuhe/Causal-Methods/bestdagsolverintheworld-main/src/exdbn"),
                Path("/Users/xiaoyuhe/Causal methods/bestdagsolverintheworld-main/src/exdbn"),
            ]
        )
        for c in candidates:
            if c.exists() and (c / "dagsolvers").exists():
                if str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                return _try_import()

        raise ModuleNotFoundError(
            "Missing optional dependency 'dagsolvers'. "
            "Either install it into this environment, or set EXDBN_DAGSOLVERS_ROOT "
            "to the folder containing the dagsolvers/ package."
        )


# ---------------------------------------------------------------------
# CaMML-style constraint helpers (ANC)
# ---------------------------------------------------------------------
def _has_path(adj: np.ndarray, source: int, dest: int) -> bool:
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[source] = True
    queue = [source]
    while queue:
        v = queue.pop(0)
        for i in range(n):
            if adj[v, i] == 1 and not visited[i]:
                visited[i] = True
                if i == dest:
                    return True
                queue.append(i)
    return visited[dest]


def write_camml_anc(
    anc_path: Path,
    *,
    n_nodes: int,
    conf: float = 0.99999,
    anc: list[tuple[int, int]] | None = None,
    forb_anc: list[tuple[int, int]] | None = None,
    abs_edges: list[tuple[int, int]] | None = None,
    labels: list[str] | None = None,
) -> Path:
    """Write a CaMML-compatible .anc file.

    - soft constraints:
      - anc:     "A => B conf;" (encourage A ancestor of B)
      - forb_anc:"A => B reconf;" where reconf = 1 - conf (discourage)
    - hard constraints:
      - abs_edges: "A -> B reconf;" (forbid edge A->B)
    """
    if labels is None:
        labels = [str(i) for i in range(n_nodes)]
    if len(labels) != n_nodes:
        raise ValueError(f"labels length {len(labels)} != n_nodes {n_nodes}")

    anc = anc or []
    forb_anc = forb_anc or []
    abs_edges = abs_edges or []

    reconf = 1.0 - float(conf)
    anc_path.parent.mkdir(parents=True, exist_ok=True)

    out_lines: list[str] = ["arcs{"]

    for v1, v2 in anc:
        out_lines.append(f"{labels[v1]} => {labels[v2]} {conf};")
    for v1, v2 in forb_anc:
        out_lines.append(f"{labels[v1]} => {labels[v2]} {reconf};")
    for v1, v2 in abs_edges:
        out_lines.append(f"{labels[v1]} -> {labels[v2]} {reconf:.5f};")

    out_lines.append("}")
    anc_path.write_text("\n".join(out_lines) + "\n")
    return anc_path


def evaluate_constraints(
    learned_adj: np.ndarray,
    *,
    anc: list[tuple[int, int]] | None = None,
    forb_anc: list[tuple[int, int]] | None = None,
    abs_edges: list[tuple[int, int]] | None = None,
) -> dict:
    """Evaluate satisfaction of hard/soft constraints on a learned adjacency."""
    anc = anc or []
    forb_anc = forb_anc or []
    abs_edges = abs_edges or []

    hard_violations = sum(int(learned_adj[i, j] == 1) for i, j in abs_edges)
    soft_anc_satisfied = sum(int(_has_path(learned_adj, i, j)) for i, j in anc)
    soft_forb_satisfied = sum(int(not _has_path(learned_adj, i, j)) for i, j in forb_anc)

    return {
        "hard_total": len(abs_edges),
        "hard_violations": hard_violations,
        "hard_satisfied": len(abs_edges) - hard_violations,
        "soft_anc_total": len(anc),
        "soft_anc_satisfied": soft_anc_satisfied,
        "soft_forb_total": len(forb_anc),
        "soft_forb_satisfied": soft_forb_satisfied,
    }


def predict_exdbn_adjacency_from_npz(
    npz_path: Path,
    *,
    sample_size: int,
    max_degree: int = 5,
) -> tuple[np.ndarray, dict]:
    """Run EXDBN (MILP) and return a binary predicted adjacency matrix.

    This is the missing key step: the hard/soft constraints should be derived
    from the EXDBN-predicted adjacency, not a random matrix.
    """
    _configure_gurobi_from_env()
    _, _, _, solve_milp = _import_dagsolvers()
    cfg = _make_milp_cfg()

    raw = np.load(npz_path)
    X_full = raw["x"]
    if X_full.shape[0] < sample_size:
        raise ValueError(f"sample_size={sample_size} exceeds rows={X_full.shape[0]}")
    X = X_full[:sample_size]

    d = X.shape[1]
    B_ref = []
    if "y" in getattr(raw, "files", []):
        try:
            B_ref = _binary_adj_from_weights(raw["y"])
        except Exception:
            B_ref = []

    W_est, _, gap, _, _ = solve_milp(
        X=X,
        cfg=cfg,
        w_threshold=0,
        Y=[],
        B_ref=B_ref,
        max_in_degree=max_degree,
        max_out_degree=max_degree,
    )
    if W_est is None:
        raise RuntimeError("EXDBN returned W_est=None")

    B_est = _binary_adj_from_weights(W_est)
    info = {
        "npz": str(npz_path),
        "rows_used": int(sample_size),
        "d": int(d),
        "max_degree": int(max_degree),
        "gap": float(gap) if gap is not None else None,
    }
    return B_est, info


def predict_exdbn_adjacency_from_csv(
    data_path: Path,
    *,
    sample_size: int | None = None,
    max_degree: int = 5,
    skiprows: int = 1,
    delimiter: str = ",",
) -> tuple[np.ndarray, dict]:
    """Run EXDBN (MILP) on a CSV dataset and return binary predicted adjacency."""
    _configure_gurobi_from_env()
    _, _, _, solve_milp = _import_dagsolvers()
    cfg = _make_milp_cfg()

    X_full = np.loadtxt(str(data_path), delimiter=delimiter, skiprows=skiprows)
    if X_full.ndim != 2:
        raise ValueError(f"Expected 2D array from {data_path}, got shape {X_full.shape}")
    if sample_size is None:
        X = X_full
    else:
        if X_full.shape[0] < sample_size:
            raise ValueError(f"sample_size={sample_size} exceeds rows={X_full.shape[0]}")
        X = X_full[:sample_size]

    d = X.shape[1]
    W_est, _, gap, _, _ = solve_milp(
        X=X,
        cfg=cfg,
        w_threshold=0,
        Y=[],
        B_ref=[],
        max_in_degree=max_degree,
        max_out_degree=max_degree,
    )
    if W_est is None:
        raise RuntimeError("EXDBN returned W_est=None")

    B_est = _binary_adj_from_weights(W_est)
    info = {
        "data": str(data_path),
        "rows_used": int(X.shape[0]),
        "d": int(d),
        "max_degree": int(max_degree),
        "gap": float(gap) if gap is not None else None,
    }
    return B_est, info


def perform_exdbn_ban_edges(
    *,
    npz_path: Path,
    out_prefix: str,
    sample_size: int,
    max_degree: int = 5,
    max_power: int = 4,
    write_anc: bool = True,
    conf: float = 0.99999,
    anc: list[tuple[int, int]] | None = None,
    forb_anc: list[tuple[int, int]] | None = None,
    labels: list[str] | None = None,
) -> dict:
    """End-to-end: EXDBN -> predicted adjacency -> hard/soft constraints files."""
    B_est, info = predict_exdbn_adjacency_from_npz(
        npz_path,
        sample_size=sample_size,
        max_degree=max_degree,
    )

    _, abs_edges = ban_edges(
        B_est,
        max_power=max_power,
        out_prefix=out_prefix,
        write_anc=write_anc,
        conf=conf,
        anc=anc,
        forb_anc=forb_anc,
        labels=labels,
    )

    metrics = evaluate_constraints(B_est, anc=anc, forb_anc=forb_anc, abs_edges=abs_edges)
    metrics.update(info)
    metrics.update(
        {
            "out_prefix": out_prefix,
            "anc_path": f"{out_prefix}.anc" if write_anc else None,
            "max_power": int(max_power),
        }
    )
    return metrics


def perform_constraints_from_data(
    *,
    data_path: Path,
    out_prefix: str,
    sample_size: int | None = None,
    max_degree: int = 5,
    max_power: int = 4,
    write_anc: bool = True,
    conf: float = 0.99999,
    anc: list[tuple[int, int]] | None = None,
    forb_anc: list[tuple[int, int]] | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Unified entry: real CSV or synthetic NPZ -> EXDBN adjacency -> constraints files."""
    data_path = Path(data_path)
    if data_path.suffix.lower() == ".npz":
        if sample_size is None:
            raise ValueError("For .npz input, sample_size must be provided")
        B_est, info = predict_exdbn_adjacency_from_npz(
            data_path,
            sample_size=sample_size,
            max_degree=max_degree,
        )
    else:
        B_est, info = predict_exdbn_adjacency_from_csv(
            data_path,
            sample_size=sample_size,
            max_degree=max_degree,
        )

    _, abs_edges = ban_edges(
        B_est,
        max_power=max_power,
        out_prefix=out_prefix,
        write_anc=write_anc,
        conf=conf,
        anc=anc,
        forb_anc=forb_anc,
        labels=labels,
    )

    metrics = evaluate_constraints(B_est, anc=anc, forb_anc=forb_anc, abs_edges=abs_edges)
    metrics.update(info)
    metrics.update(
        {
            "out_prefix": out_prefix,
            "anc_path": f"{out_prefix}.anc" if write_anc else None,
            "max_power": int(max_power),
        }
    )
    return metrics

# ---------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------
def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v in (None, "") else v

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v in (None, "") else int(v)

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v in (None, "") else float(v)


def _configure_gurobi_from_env() -> None:
    """Best-effort: configure Gurobi global parameters for progress output.

    This helps when EXDBN (MILP) feels "stuck" with no visible output.
    All params are optional; if gurobipy isn't available, this is a no-op.

    Supported env vars:
    - EXDBN_GUROBI_OUTPUTFLAG (default: 1)
    - EXDBN_GUROBI_LOGFILE (default: "")
    - EXDBN_GUROBI_THREADS (default: "")
    - EXDBN_GUROBI_DISPLAYINTERVAL (default: "")
    """
    try:
        import gurobipy as gp
    except Exception:
        return

    output_flag = os.getenv("EXDBN_GUROBI_OUTPUTFLAG", "1")
    log_file = os.getenv("EXDBN_GUROBI_LOGFILE", "")
    threads = os.getenv("EXDBN_GUROBI_THREADS", "")
    display_interval = os.getenv("EXDBN_GUROBI_DISPLAYINTERVAL", "")

    try:
        gp.setParam("OutputFlag", int(output_flag))
    except Exception:
        pass
    if log_file:
        try:
            gp.setParam("LogFile", log_file)
        except Exception:
            pass
    if threads:
        try:
            gp.setParam("Threads", int(threads))
        except Exception:
            pass
    if display_interval:
        try:
            gp.setParam("DisplayInterval", int(display_interval))
        except Exception:
            pass

def _env_int_list(name: str, default_csv: str) -> list[int]:
    raw = _env_str(name, default_csv)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _binary_adj_from_weights(W: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    return (np.abs(W) > threshold).astype(int)

def _ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_dir = out_dir / "edges"
    edges_dir.mkdir(exist_ok=True)
    return out_dir, edges_dir

# ---------------------------------------------------------------------
# MILP config
# ---------------------------------------------------------------------
def _make_milp_cfg():
    cfg = OmegaConf.create({})
    cfg.time_limit = _env_int("EXDBN_TIME_LIMIT", 18000)
    cfg.constraints_mode = _env_str("EXDBN_CONSTRAINTS_MODE", "weights")
    cfg.callback_mode = _env_str("EXDBN_CALLBACK_MODE", "all_cycles")
    cfg.lambda1 = _env_float("EXDBN_LAMBDA1", 1.0)
    cfg.lambda2 = _env_float("EXDBN_LAMBDA2", 1.0)
    cfg.loss_type = _env_str("EXDBN_LOSS_TYPE", "l2")
    cfg.reg_type = _env_str("EXDBN_REG_TYPE", "l1")
    cfg.a_reg_type = _env_str("EXDBN_A_REG_TYPE", "l1")
    cfg.robust = _env_str("EXDBN_ROBUST", "0") == "1"
    cfg.weights_bound = _env_float("EXDBN_WEIGHTS_BOUND", 100.0)
    cfg.target_mip_gap = _env_float("EXDBN_TARGET_MIP_GAP", 0.001)
    cfg.problem_name = _env_str("EXDBN_PROBLEM_NAME", "er")
    return cfg

# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------
def generate_datasets(out_dir: Path, is_dynamic: bool):
    """
    Generate synthetic datasets with flexible features and variant.
    Static: ER/SF graph
    Dynamic: DBN
    """

    # 从环境变量读取参数
    features_list = _env_int_list("EXDBN_GEN_FEATURES", "5,10,15,20,25,30,35")
    n = _env_int("EXDBN_GEN_NUMBER_OF_SAMPLES", 2000)
    variant = _env_str("EXDBN_GEN_VARIANT", "er")       # graph type for load_problem
    sem_type = _env_str("EXDBN_GEN_SEM_TYPE", "gauss")  # 静态图用
    generator = _env_str("EXDBN_GEN_GENERATOR", "notears")  # 动态图用

    # 文件名前缀
    file_prefix = "dynamic" if is_dynamic else "static"

    out_dir.mkdir(parents=True, exist_ok=True)

    for d in features_list:
        print(f"[GEN] {file_prefix}: d={d}, n={n}")

        load_problem_dict, _, _, _ = _import_dagsolvers()

        # 这里使用 variant 作为 load_problem 的 name，保证不会报 unknown problem
        problem_cfg = {
            "name": variant,              # er / sf / 其他支持类型
            "number_of_variables": d,
            "number_of_samples": n,
            "noise_scale": 1.0,
            "sem_type": sem_type,
            'edge_ratio': 0.5 
        }

        if is_dynamic:
            # 动态 DBN 配置
            problem_cfg.update({
                "p": 2,
                "generator": generator,
                "graph_type_intra": variant,
                "graph_type_inter": variant,
                "intra_edge_ratio": 0.5,
                "inter_edge_ratio": 0.5,
                "w_max_intra": 1.0,
                "w_min_intra": 0.01,
                "w_max_inter": 0.2,
                "w_min_inter": 0.01,
                "w_decay": 1.0,
            })
        else:
            # 静态图配置
            problem_cfg.update({
                "p": 0,
            })

        cfg = OmegaConf.create({"problem": problem_cfg})
        problem = load_problem_dict(cfg)

        # 输出文件名用 static/dynamic + variant + sem_type + 维度
        np.savez(
            out_dir / f"{file_prefix}_{variant}_{sem_type}_{d}.npz",
            x=problem["X"],
            y=problem["W_true"],
        )


def generate_all_datasets():
    base = Path(_env_str("EXDBN_DATA_DIR", "datasets/syntheticdata"))
    generate_datasets(base / "static", is_dynamic=False)
    generate_datasets(base / "dynamic", is_dynamic=True)

# ---------------------------------------------------------------------
# EXDBN runner
# ---------------------------------------------------------------------
def run_exdbn_single(npz_path: Path, out_dir: Path, sample_size: int) -> None:
    _, _, count_accuracy, solve_milp = _import_dagsolvers()
    cfg = _make_milp_cfg()
    out_dir, edges_dir = _ensure_dirs(out_dir)

    raw = np.load(npz_path)
    X_full = raw["x"]
    W_true = raw["y"]
    d = X_full.shape[1]

    if X_full.shape[0] < sample_size:
        return

    X = X_full[:sample_size]
    B_true = _binary_adj_from_weights(W_true)

    runtime_file = out_dir / f"runtime_{npz_path.stem}_n{sample_size}.csv"
    columns = [
        "dataset", "features", "samples", "degree", "runtime_seconds",
        "gap", "fdr", "tpr", "fpr", "shd", "nnz", "precision", "f1score", "g_score"
    ]
    if not runtime_file.exists() or _env_str("EXDBN_OVERWRITE", "0") == "1":
        pd.DataFrame(columns=columns).to_csv(runtime_file, index=False)

    degrees = _env_int_list("EXDBN_MAX_DEGREES", "5")

    for deg in degrees:
        start = time.time()
        W_est, _, gap, _, _ = solve_milp(
            X=X,
            cfg=cfg,
            w_threshold=0,
            Y=[],
            B_ref=B_true,
            max_in_degree=deg,
            max_out_degree=deg,
        )
        elapsed = time.time() - start
        if W_est is None:
            break

        B_est = _binary_adj_from_weights(W_est)
        metrics_full = count_accuracy(B_true, B_est, [], [], test_dag=True)
        metrics = {k: metrics_full.get(k, np.nan) for k in columns if k not in ["dataset", "features", "samples", "degree", "runtime_seconds", "gap"]}
        print(f"[EXDBN] {npz_path.stem} d={d} n={sample_size} deg={deg} time={elapsed:.2f}s gap={gap:.4f} metrics={metrics}")

        row = {
            "dataset": npz_path.stem,
            "features": d,
            "samples": sample_size,
            "degree": deg,
            "runtime_seconds": elapsed,
            "gap": gap,
            **metrics,
        }
        pd.DataFrame([row])[columns].to_csv(runtime_file, mode="a", header=False, index=False)

def run_exdbn_parallel():
    base_data = Path(_env_str("EXDBN_DATA_DIR", "datasets/syntheticdata"))
    base_out = Path(_env_str("EXDBN_OUT_DIR", "results/exdbn"))
    sample_sizes = _env_int_list("EXDBN_SAMPLE_SIZES", "2000")
    max_workers = _env_int("EXDBN_NUM_WORKERS", os.cpu_count() // 2)

    tasks = []
    for mode in ["static", "dynamic"]:
        for npz in (base_data / mode).glob("*.npz"):
            for n in sample_sizes:
                out_dir = base_out / f"{mode}_n{n}"
                tasks.append((npz, out_dir, n))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_exdbn_single, npz, out, n) for npz, out, n in tasks]
        for f in as_completed(futures):
            f.result()

def ban_edges(
    adj: np.ndarray,
    max_power: int = 4,
    out_prefix: str | None = None,
    *,
    write_anc: bool = False,
    conf: float = 0.99999,
    anc: list[tuple[int, int]] | None = None,
    forb_anc: list[tuple[int, int]] | None = None,
    labels: list[str] | None = None,
):
    """
    计算可达性互补矩阵，并自动写入hard constraints和prior constraints格式文件。
    adj: 原始邻接矩阵 (0/1)
    max_power: 可达性最大幂次
    out_prefix: 输出文件前缀（如指定则写文件）
    返回: 互补矩阵, banned_edges列表
    """
    n = adj.shape[0]
    reach = np.zeros_like(adj)
    power = np.eye(n, dtype=int)
    for _ in range(1, max_power + 1):
        power = np.matmul(power, adj)
        reach = ((reach + power) > 0).astype(int)
    # 加上直接边
    reach = ((reach + adj) > 0).astype(int)
    complement = 1 - reach
    np.fill_diagonal(complement, 0)
    banned_edges = [(i, j) for i in range(n) for j in range(n) if complement[i, j] == 1]

    if out_prefix:
        prefix_path = Path(out_prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        # 写 hard constraints 文件
        with open(f"{prefix_path}_hard_constraints.txt", "w") as f:
            for i, j in banned_edges:
                f.write(f"{i} {j}\n")
        # 写 prior constraints 文件（如需要可自定义格式）
        with open(f"{prefix_path}_prior_constraints.txt", "w") as f:
            for i, j in banned_edges:
                f.write(f"forbid: {i} -> {j}\n")

        # 可选：写 CaMML 兼容的 .anc 文件（hard/soft 约束）
        if write_anc:
            anc_path = Path(f"{prefix_path}.anc")
            write_camml_anc(
                anc_path,
                n_nodes=n,
                conf=conf,
                anc=anc,
                forb_anc=forb_anc,
                abs_edges=banned_edges,
                labels=labels,
            )
    return complement, banned_edges

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if _env_str("EXDBN_GENERATE_DYNAMIC", "0") == "1":
        base_dir = Path("datasets/syntheticdata")
        generate_datasets(base_dir / "static", is_dynamic=False)
        generate_datasets(base_dir / "dynamic", is_dynamic=True)
        return

    print("Running EXDBN in parallel...")
    run_exdbn_parallel()
    print("=== Done ===")

if __name__ == "__main__":
    # 示例：生成一个随机邻接矩阵（可替换为实际ExDBN/LLM输出）
    n = 5  # 节点数，可根据实际需求调整
    np.random.seed(42)
    adj = (np.random.rand(n, n) > 0.7).astype(int)
    np.fill_diagonal(adj, 0)
    print("原始邻接矩阵：\n", adj)

    # 计算并输出ExDBN+LLM约束
    complement, banned_edges = ban_edges(adj, max_power=4, out_prefix="output/ExDBN_LLM")
    print("互补矩阵：\n", complement)
    print(f"已写入 hard constraints 和 prior constraints 文件，边数: {len(banned_edges)}")
