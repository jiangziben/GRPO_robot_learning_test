"""
GRPO (Group Relative Policy Optimization) 算法封装。

- GRPO 类：纯算法（advantage 计算 + PPO 截断更新），不依赖环境和模型

支持离散动作（Categorical）和连续动作（Normal）两种策略类型。
"""

import torch
from torch.distributions import Normal


# ==========================================================================
# GRPO 算法类
# ==========================================================================

class GRPO:
    """GRPO 算法封装类（纯算法，不依赖环境和模型）。

    Args:
        optimizer:   PyTorch 优化器
        eps:         PPO 截断参数
        n_iterations: 每次 update 的更新轮数
        discrete:    True=离散动作，False=连续动作
    """

    def __init__(self, optimizer, eps=0.2, n_iterations=20, discrete=True,
                 entropy_coef=0.0):
        self.optimizer = optimizer
        self.eps = eps
        self.n_iterations = n_iterations
        self.discrete = discrete
        self.entropy_coef = entropy_coef

    # ------------------------------------------------------------------
    # Advantage 计算
    # ------------------------------------------------------------------
    def compute_advantages(self, trajectories):
        """计算标准化 advantage（GRPO 核心：组内相对奖励标准化）。

        Args:
            trajectories: dict，包含 "normalized_rewards"，shape [batch_size]

        Returns:
            advantages: shape [batch_size]
        """
        rewards = trajectories["normalized_rewards"]
        mean = torch.mean(rewards)
        std = torch.std(rewards) + 1e-8
        return (rewards - mean) / std

    def update(self, policy, trajectories):
        """使用 PPO 截断损失更新策略网络。

        Args:
            policy:       策略网络（nn.Module）
            trajectories: collect_trajectories 返回的 dict

        Returns:
            loss: 最后一轮更新的损失值
        """
        advantages = self.compute_advantages(trajectories).unsqueeze(-1)  # [B, 1]
        all_states = trajectories["all_states"]
        all_log_probs = trajectories["all_log_probs"]
        all_actions = trajectories["all_actions"]

        for _ in range(self.n_iterations):
            if self.discrete:
                probs = policy(all_states)                                        # [B, T, n_actions]
                dist = torch.distributions.Categorical(probs)
                new_log_probs = torch.log(probs.gather(-1, all_actions.unsqueeze(-1)))  # [B, T, 1]
                old_log_probs = all_log_probs.unsqueeze(-1)                      # [B, T, 1]
            else:
                mu, sigma = policy(all_states)                                    # [B, T, act_dim]
                dist = Normal(mu, sigma)
                new_log_probs = dist.log_prob(all_actions)                       # [B, T, act_dim]
                old_log_probs = all_log_probs

            ratio = torch.exp(new_log_probs - old_log_probs)                     # [B, T, 1]
            adv = advantages.unsqueeze(1)                                        # [B, 1, 1]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv
            entropy = dist.entropy().mean()                                      # 熵正则
            loss = torch.mean(-torch.min(surr1, surr2)) - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()
