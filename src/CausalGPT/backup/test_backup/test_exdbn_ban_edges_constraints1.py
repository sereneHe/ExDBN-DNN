"""测试ExDBN约束评估功能。"""

import numpy as np
from exdbn_ban_edges import evaluate_exdbn_constraints

def ExDBN_evaluation(ev_dag, anc, forb_anc):
    """
    评估ev_dag是否满足anc/forb_anc约束。
    ev_dag: 推断出的邻接矩阵 (0/1)
    anc, forb_anc: 约束列表
    返回满足的数量
    """
    return evaluate_exdbn_constraints(ev_dag, anc, forb_anc)

if __name__ == "__main__":
    # 示例：生成一个随机推断邻接矩阵和约束
    n = 5
    np.random.seed(123)
    ev_dag = (np.random.rand(n, n) > 0.7).astype(int)
    np.fill_diagonal(ev_dag, 0)
    anc = [(0, 1), (2, 3)]
    forb_anc = [(1, 2), (3, 4)]
    num_satisfy = ExDBN_evaluation(ev_dag, anc, forb_anc)
    print(f"满足的约束数量: {num_satisfy}")
