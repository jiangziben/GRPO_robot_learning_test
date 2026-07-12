"""连续动作策略网络（Pendulum 等）。"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal


class PolicyNetContinuous(torch.nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc_mu = torch.nn.Linear(128, action_dim)
        self.fc_std = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)).clamp(0.05, 1.0)
        return mu, std

    def sample_with_log_prob(self, state_tensor):
        """训练用：从当前策略采样动作并返回 log_prob。

        Args:
            state_tensor: [B, state_dim] 已在目标设备上

        Returns:
            actions:   [B, act_dim]
            log_probs: [B, act_dim] (detached)
        """
        mu, sigma = self.forward(state_tensor)
        dist = Normal(mu, sigma)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).detach()
        return actions, log_probs

    def take_action(self, state, deterministic=False):
        """推理用：单个状态 → 单个动作。"""
        device = next(self.parameters()).device
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, sigma = self.forward(state_tensor)
            if deterministic:
                action = mu
            else:
                action = Normal(mu, sigma).sample()
            return action.cpu().numpy()[0]
