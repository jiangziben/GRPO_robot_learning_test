"""Value 网络（用于 PPO 等需要 Critic 的算法）。"""

import torch
import torch.nn.functional as F


class ValueNet(torch.nn.Module):
    """状态价值估计网络。输入 state，输出 scalar V(s)。"""

    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
