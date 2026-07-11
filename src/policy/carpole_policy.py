"""离散动作策略网络（CartPole 等）。"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNet(torch.nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)

    def sample_with_log_prob(self, state_tensor):
        """训练用：从当前策略采样动作并返回 log_prob。

        Args:
            state_tensor: [B, state_dim] 已在目标设备上

        Returns:
            actions:   [B]
            log_probs: [B] (detached)
        """
        probs = self.forward(state_tensor)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).detach()
        return actions, log_probs

    def take_action(self, state, deterministic=False):
        """推理用：单个状态 → 单个动作。"""
        device = next(self.parameters()).device
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = self.forward(state_tensor)
            if deterministic:
                return torch.argmax(probs, dim=-1).item()
            return Categorical(probs).sample().item()
