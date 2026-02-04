import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def parse_anc_file(anc_path, n):
    mask = np.zeros((n, n), dtype=int)
    with open(anc_path, "r") as f:
        for line in f:
            if "->" in line:
                parts = line.strip().replace(';','').replace('}','').split()
                if len(parts) >= 3:
                    i, j = int(parts[0]), int(parts[2])
                    mask[i, j] = 1
    return mask

class DeepDNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_dnn_with_constraints(n, mask, epochs=300):
    model = DeepDNN(n)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        x = torch.eye(n)
        pred = model(x)
        pred = pred * (1 - torch.tensor(mask, dtype=torch.float32))
        # 这里loss可自定义，假设我们希望输出接近0（无结构），仅作演示
        loss = pred.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (pred.detach().numpy() > 0.5).astype(int)

if __name__ == "__main__":
    n = 5
    mask = parse_anc_file("../../test_ancs.anc", n)
    adj_dnn = train_dnn_with_constraints(n, mask, epochs=300)
    print("DNN结构预测：\n", adj_dnn)

    # 假设 exdbn 结构如下（实际应为 exdbn 推断结果）
    adj_exdbn = np.random.randint(0, 2, (n, n))
    np.fill_diagonal(adj_exdbn, 0)

    # 评估
    try:
        from exdbn_ban_edges import evaluate_exdbn_constraints
        anc = [(0, 1), (2, 3)]
        forb_anc = [(1, 2)]
        print("exdbn 满足约束数：", evaluate_exdbn_constraints(adj_exdbn, anc, forb_anc))
        print("exdbn+DNN 满足约束数：", evaluate_exdbn_constraints(adj_dnn, anc, forb_anc))
    except ImportError:
        print("请确保 exdbn_ban_edges.py 中有 evaluate_exdbn_constraints 函数并在 PYTHONPATH 下可用。")
