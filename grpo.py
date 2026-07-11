"""
GRPO (Group Relative Policy Optimization) 算法封装。

- GRPO 类：纯算法（advantage 计算 + PPO 截断更新 + 推理），不依赖环境
- collect_trajectories：轨迹收集工具函数

支持离散动作（Categorical）和连续动作（Normal）两种策略类型。
"""

import numpy as np
import torch
from torch.distributions import Categorical, Normal


# ==========================================================================
# 工具函数
# ==========================================================================

def collect_trajectories(envs, policy, num_steps=500, discrete=True,
                         device="cpu", reward_fn=None):
    """从 vectorized 环境并行收集轨迹。

    Args:
        envs:       Gym vectorized 环境
        policy:     策略网络
        num_steps:  每条轨迹最大步数
        discrete:   True=离散动作，False=连续动作
        device:     计算设备
        reward_fn:  环境特定的奖励变换函数，
                    签名: fn(next_states, rewards, dones, all_dones) -> rewards

    Returns:
        trajectories:   dict，包含 all_states, all_log_probs, all_actions, normalized_rewards
        episode_rewards: [group_size] 每个并行环境的原始累计奖励
    """
    group_size = envs.num_envs
    seed_num = np.random.randint(0, 1000)
    states = envs.reset(seed=[seed_num] * group_size)

    all_states = []
    all_actions = []
    all_log_probs = []
    all_rewards = torch.zeros(group_size)
    all_dones = torch.tensor([False] * group_size)

    for _ in range(num_steps):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)

        if discrete:
            probs = policy(states_tensor)
            dist = Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).detach()
        else:
            mu, sigma = policy(states_tensor)
            dist = Normal(mu, sigma)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).detach()

        next_states, rewards, dones, infos = envs.step(actions.cpu().numpy())

        all_states.append(states)
        all_actions.append(actions)
        all_log_probs.append(log_probs)

        all_dones[dones] = True
        if reward_fn is not None:
            rewards = reward_fn(next_states, rewards, dones, all_dones)
        all_rewards += rewards

        states = next_states
        if torch.all(all_dones):
            break

    normalized_rewards = (all_rewards / num_steps).to(device)
    all_states = torch.tensor(np.array(all_states), dtype=torch.float32,
                              device=device).permute(1, 0, 2)
    if discrete:
        all_log_probs = torch.stack(all_log_probs).permute(1, 0).to(device)
        all_actions = torch.stack(all_actions).permute(1, 0).to(device)
    else:
        all_log_probs = torch.stack(all_log_probs).permute(1, 0, 2).to(device)
        all_actions = torch.stack(all_actions).permute(1, 0, 2).to(device)

    trajectories = {
        "all_states": all_states,
        "all_log_probs": all_log_probs,
        "all_actions": all_actions,
        "normalized_rewards": normalized_rewards
    }
    episode_rewards = normalized_rewards * num_steps
    return trajectories, episode_rewards


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

    def __init__(self, optimizer, eps=0.2, n_iterations=20, discrete=True):
        self.optimizer = optimizer
        self.eps = eps
        self.n_iterations = n_iterations
        self.discrete = discrete

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
            loss = torch.mean(-torch.min(surr1, surr2))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()
