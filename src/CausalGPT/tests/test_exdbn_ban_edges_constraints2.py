import numpy as np
from exdbn_ban_edges import perform_exdbn_with_constraints

def ExDBN_perform(adj, max_power=4, conf=0.99999, out_prefix="output/exdbn_llm"):
    """
    生成ExDBN+LLM风格的hard/soft约束anc文件。
    adj: 邻接矩阵 (0/1)
    max_power: 可达性最大幂次
    conf: soft约束置信度
    out_prefix: 输出文件前缀
    返回: abs_edges, anc, forb_anc
    """
    abs_edges, anc, forb_anc = perform_exdbn_with_constraints(adj, max_power=max_power, conf=conf, out_prefix=out_prefix)
    return abs_edges, anc, forb_anc

if __name__ == "__main__":
    # 示例：生成一个随机邻接矩阵
    n = 5
    np.random.seed(42)
    adj = (np.random.rand(n, n) > 0.7).astype(int)
    np.fill_diagonal(adj, 0)
    abs_edges, anc, forb_anc = ExDBN_perform(adj, max_power=4, conf=0.99999, out_prefix="output/exdbn_llm")
    print("abs_edges:", abs_edges)
    print("anc:", anc)
    print("forb_anc:", forb_anc)
import numpy as np
from exdbn_ban_edges import perform_exdbn_with_constraints

def ExDBN_perform(adj, max_power=4, conf=0.99999, out_prefix="output/exdbn_llm"):
    """
    生成ExDBN+LLM风格的hard/soft约束anc文件。
    adj: 邻接矩阵 (0/1)
    max_power: 可达性最大幂次
    conf: soft约束置信度
    out_prefix: 输出文件前缀
    返回: abs_edges, anc, forb_anc
    """
    abs_edges, anc, forb_anc = perform_exdbn_with_constraints(adj, max_power=max_power, conf=conf, out_prefix=out_prefix)
    return abs_edges, anc, forb_anc

if __name__ == "__main__":
    # 示例：生成一个随机邻接矩阵
    n = 5
    np.random.seed(42)
    adj = (np.random.rand(n, n) > 0.7).astype(int)
    np.fill_diagonal(adj, 0)
    abs_edges, anc, forb_anc = ExDBN_perform(adj, max_power=4, conf=0.99999, out_prefix="output/exdbn_llm")
    print("abs_edges:", abs_edges)
    print("anc:", anc)
    print("forb_anc:", forb_anc)
