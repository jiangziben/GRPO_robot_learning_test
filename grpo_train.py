"""GRPO 统一训练脚本。支持 CartPole 和 Pendulum 环境。

用法:
    python grpo_train.py --env cartpole
    python grpo_train.py --env pendulum
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import gym

from grpo import GRPO, collect_trajectories
from model import PolicyNet, PolicyNetContinuous


# ---------------------------------------------------------------------------
# 环境配置
# ---------------------------------------------------------------------------
ENV_CONFIGS = {
    "cartpole": {
        "env_name": "CartPole-v1",
        "policy_cls": PolicyNet,
        "discrete": True,
        "group_size": 20,
        "iteration_num": 100,
        "max_steps": 500,
        "lr": 0.02,
        "reward_fn": lambda ns, r, _d, ad: _cartpole_reward(ns, r, ad),
    },
    "pendulum": {
        "env_name": "Pendulum-v1",
        "policy_cls": PolicyNetContinuous,
        "discrete": False,
        "group_size": 100,
        "iteration_num": 500,
        "max_steps": 500,
        "lr": 0.001,
        "reward_fn": lambda ns, r, _d, _ad: r + ns[:, 0],
    },
}


def _cartpole_reward(next_states, rewards, all_dones):
    rewards = rewards.copy()
    rewards[all_dones.numpy()] = 0
    rewards += -np.abs(next_states[:, 0])
    return rewards


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------
def train(env_name: str):
    cfg = ENV_CONFIGS[env_name]

    print(f"Training GRPO on {cfg['env_name']}")
    print(f"  group_size={cfg['group_size']}, iterations={cfg['iteration_num']}, "
          f"max_steps={cfg['max_steps']}, lr={cfg['lr']}")

    envs = gym.vector.make(cfg["env_name"], num_envs=cfg["group_size"])
    state_dim = envs.single_observation_space.shape[0]
    n_actions = (envs.single_action_space.n if cfg["discrete"]
                 else envs.single_action_space.shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = cfg["policy_cls"](state_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    grpo = GRPO(optimizer, eps=0.2, n_iterations=20, discrete=cfg["discrete"])

    return_list = []
    start = time.time()

    for i_iter in tqdm(range(cfg["iteration_num"])):
        trajectories, episode_rewards = collect_trajectories(
            envs, policy, cfg["max_steps"],
            discrete=cfg["discrete"], device=device,
            reward_fn=cfg["reward_fn"],
        )
        loss = grpo.update(policy, trajectories)

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        return_list.append(avg_reward.cpu().numpy())

        if i_iter != 0 and i_iter % 200 == 0:
            path = f"./weights/grpo_{env_name}_update_{i_iter}.pth"
            torch.save(policy.state_dict(), path)

        print(f"iter {i_iter:4d}, avg_reward: {avg_reward:.2f}, loss: {loss:.4f}")

    print(f"used_time(s): {time.time() - start:.1f}")

    save_path = f"./weights/grpo_{env_name}_policy_final.pth"
    torch.save(policy.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    envs.close()

    # 绘图
    plt.plot(range(len(return_list)), return_list)
    plt.xlabel("Iterations")
    plt.ylabel("Returns")
    plt.title(f"GRPO on {cfg['env_name']}")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "pendulum"], required=True)
    args = parser.parse_args()
    train(args.env)
