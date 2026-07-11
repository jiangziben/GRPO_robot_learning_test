import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNet(torch.nn.Module):
    """离散动作策略网络（用于 CartPole 等离散环境）。"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)

    def take_action(self, state, deterministic=False):
        """单步动作推理。state: 单个环境状态 [state_dim]"""
        device = next(self.parameters()).device
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = self.forward(state_tensor)
            if deterministic:
                return torch.argmax(probs, dim=-1).item()
            else:
                return Categorical(probs).sample().item()
