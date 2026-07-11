"""
PPO (Proximal Policy Optimization) 算法封装。

- PPO 类：纯算法（GAE advantage + PPO 截断更新），不依赖环境和模型
- 支持离散动作和连续动作
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal


class PPO:
    """PPO 算法封装类（纯算法，不依赖环境和模型）。

    Args:
        actor_optimizer:   Actor 优化器
        critic_optimizer:  Critic 优化器
        eps:               PPO 截断参数
        gamma:             折扣因子
        lmbda:             GAE λ 参数
        n_iterations:      每次 update 的更新轮数
        discrete:          True=离散动作，False=连续动作
    """

    def __init__(self, actor_optimizer, critic_optimizer,
                 eps=0.2, gamma=0.98, lmbda=0.95,
                 n_iterations=10, discrete=True):
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.eps = eps
        self.gamma = gamma
        self.lmbda = lmbda
        self.n_iterations = n_iterations
        self.discrete = discrete

    # ------------------------------------------------------------------
    # GAE 计算
    # ------------------------------------------------------------------
    def compute_gae(self, rewards, values, dones):
        """计算 GAE advantages 和 returns。

        Args:
            rewards:  [B, T] 每步奖励
            values:   [B, T] 每步的状态价值 V(s_t)
            dones:    [B, T] 每步的终止标志（1=终止, 0=未终止）

        Returns:
            advantages: [B, T]
            returns:    [B, T]  TD(λ) target
        """
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=rewards.device)

        # 从后向前计算 GAE
        for t in reversed(range(T)):
            next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=rewards.device)
            next_non_terminal = 1.0 - dones[:, t]
            delta = rewards[:, t] + self.gamma * next_value * next_non_terminal - values[:, t]
            gae = delta + self.gamma * self.lmbda * next_non_terminal * gae
            advantages[:, t] = gae

        returns = advantages + values[:, :T]
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO 更新
    # ------------------------------------------------------------------
    def update(self, actor, critic, trajectories):
        """使用 PPO 截断损失更新 actor 和 critic。

        Args:
            actor:        策略网络（nn.Module）
            critic:       价值网络（nn.Module），输出 V(s)
            trajectories: collect_ppo_trajectories 返回的 dict

        Returns:
            actor_loss:  最后一轮 actor 损失
            critic_loss: 最后一轮 critic 损失
        """
        all_states = trajectories["all_states"]              # [B, T, state_dim]
        all_next_states = trajectories["all_next_states"]    # [B, T, state_dim]
        all_log_probs = trajectories["all_log_probs"]        # [B, T] or [B, T, act_dim]
        all_actions = trajectories["all_actions"]            # [B, T] or [B, T, act_dim]
        all_rewards = trajectories["all_rewards"]            # [B, T]
        all_dones = trajectories["all_dones"]                # [B, T]

        # 计算 values（no_grad：GAE 用旧 critic 值，不参与当前计算图）
        with torch.no_grad():
            values = critic(all_states).squeeze(-1)              # [B, T]
            next_values = critic(all_next_states).squeeze(-1)    # [B, T]

        # 拼接 values 用于 GAE（需要 V(s_T) 作为最后一步的 bootstrap）
        values_with_bootstrap = torch.cat([values, next_values[:, -1:]], dim=1)  # [B, T+1]

        advantages, returns = self.compute_gae(all_rewards, values_with_bootstrap, all_dones)
        # advantages, returns: [B, T]

        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = None
        critic_loss = None

        for _ in range(self.n_iterations):
            if self.discrete:
                probs = actor(all_states)                                             # [B, T, n_actions]
                new_log_probs = torch.log(probs.gather(-1, all_actions.unsqueeze(-1)))  # [B, T, 1]
                old_log_probs = all_log_probs.unsqueeze(-1)                            # [B, T, 1]
            else:
                mu, sigma = actor(all_states)                                          # [B, T, act_dim]
                dist = Normal(mu, sigma)
                new_log_probs = dist.log_prob(all_actions)                             # [B, T, act_dim]
                old_log_probs = all_log_probs

            ratio = torch.exp(new_log_probs - old_log_probs)                           # [B, T, 1]
            adv = advantages.unsqueeze(-1)                                             # [B, T, 1]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            # Critic loss
            current_values = critic(all_states).squeeze(-1)                            # [B, T]
            critic_loss = F.mse_loss(current_values, returns.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()
