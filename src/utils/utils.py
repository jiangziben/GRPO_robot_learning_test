"""轨迹收集工具函数。"""

import numpy as np
import torch
from torch.distributions import Categorical, Normal


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
