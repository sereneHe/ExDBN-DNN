import numpy as np

def write_anc_file(n, abs_edges=None, anc=None, forb_anc=None, conf=0.99999, out_path="test_ancs.anc"):
    if abs_edges is None:
        abs_edges = []
    if anc is None:
        anc = []
    if forb_anc is None:
        forb_anc = []
    reconf = 1 - conf
    with open(out_path, "w") as ofh:
        ofh.write("arcs{\n")
        for v1, v2 in anc:
            ofh.write(f"{v1} => {v2} {conf};\n")
        for v1, v2 in forb_anc:
            ofh.write(f"{v1} => {v2} {reconf};\n")
        for v1, v2 in abs_edges:
            ofh.write(f"{v1} -> {v2} {reconf:.5f};\n")
        ofh.write("}")
    print(f"已写入约束文件: {out_path}")

if __name__ == "__main__":
    n = 5
    np.random.seed(42)
    adj = (np.random.rand(n, n) > 0.7).astype(int)
    np.fill_diagonal(adj, 0)
    abs_edges = [(i, j) for i in range(n) for j in range(n) if adj[i, j] == 0]
    anc = [(0, 1), (2, 3)]
    forb_anc = [(1, 2)]
    write_anc_file(n, abs_edges=abs_edges, anc=anc, forb_anc=forb_anc, conf=0.99999, out_path="test_ancs.anc")
