"""统一训练脚本。支持 GRPO 和 PPO，支持 CartPole 和 Pendulum 环境。

配置文件（含 algo、env 及所有超参）放在 config/ 目录下。

用法:
    python train.py --config grpo_cartpole.json
    python train.py --config ppo_pendulum.json
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import gym

from src.rl.grpo import GRPO
from src.rl.ppo import PPO
from src.utils.utils import collect_trajectories, collect_ppo_trajectories
from src.model.carpole_policy import PolicyNet
from src.model.pendulum_policy import PolicyNetContinuous
from src.model.value_net import ValueNet


# ---------------------------------------------------------------------------
# 环境元信息（结构性配置，不适合放 JSON）
# ---------------------------------------------------------------------------
ENV_META = {
    "cartpole": {
        "env_name": "CartPole-v1",
        "policy_cls": PolicyNet,
        "discrete": True,
        "reward_fn": lambda ns, r, _d, ad: _cartpole_reward(ns, r, ad),
    },
    "pendulum": {
        "env_name": "Pendulum-v1",
        "policy_cls": PolicyNetContinuous,
        "discrete": False,
        "reward_fn": {
            "grpo": lambda ns, r, _d, _ad: r + ns[:, 0],
            "ppo":  lambda ns, r, _d, _ad: (r + 8.0) / 8.0,
        },
    },
}


def _cartpole_reward(next_states, rewards, all_dones):
    rewards = rewards.copy()
    rewards[all_dones.numpy()] = 0
    rewards += -np.abs(next_states[:, 0])
    return rewards


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_config(config_file: str) -> dict:
    """读取配置文件。若为相对/绝对路径则直接使用，否则在 config/ 目录下查找。"""
    if os.path.sep in config_file or os.path.altsep in config_file:
        path = config_file  # 已包含路径
    else:
        path = os.path.join(CONFIG_DIR, config_file)
    return load_json(path)


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------
def train(algo: str, env_name: str, cfg: dict):
    meta = ENV_META[env_name]

    print(f"Training {algo.upper()} on {meta['env_name']}")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    discrete = meta["discrete"]
    num_envs = cfg["num_envs"]
    max_steps = cfg["max_steps"]
    iteration_num = cfg["iteration_num"]

    envs = gym.vector.make(meta["env_name"], num_envs=num_envs)
    state_dim = envs.single_observation_space.shape[0]
    n_actions = (envs.single_action_space.n if discrete
                 else envs.single_action_space.shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = meta["policy_cls"](state_dim, n_actions).to(device)

    reward_fn = meta["reward_fn"]
    if isinstance(reward_fn, dict):
        reward_fn = reward_fn[algo]

    if algo == "grpo":
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
        algo_inst = GRPO(optimizer, eps=cfg["eps"],
                         n_iterations=cfg["n_iterations"],
                         discrete=discrete)
    else:  # ppo
        critic = ValueNet(state_dim).to(device)
        actor_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["actor_lr"])
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg["critic_lr"])
        algo_inst = PPO(actor_optimizer, critic_optimizer,
                        eps=cfg["eps"], gamma=cfg["gamma"],
                        lmbda=cfg["lmbda"],
                        n_iterations=cfg["n_iterations"],
                        discrete=discrete)

    return_list = []
    start = time.time()

    for i_iter in tqdm(range(iteration_num)):
        if algo == "grpo":
            trajectories, episode_rewards = collect_trajectories(
                envs, policy, max_steps,
                discrete=discrete, device=device,
                reward_fn=reward_fn,
            )
            loss = algo_inst.update(policy, trajectories)
            log_str = f"loss: {loss:.4f}"
        else:  # ppo
            trajectories, episode_rewards = collect_ppo_trajectories(
                envs, policy, max_steps,
                discrete=discrete, device=device,
                reward_fn=reward_fn,
            )
            actor_loss, critic_loss = algo_inst.update(policy, critic, trajectories)
            log_str = f"actor_loss: {actor_loss:.4f}, critic_loss: {critic_loss:.4f}"

        avg_reward = (sum(episode_rewards) / len(episode_rewards)).item()
        return_list.append(avg_reward)

        if i_iter != 0 and i_iter % 200 == 0:
            base = os.path.splitext(cfg["save_path"])[0]
            path = f"{base}_update_{i_iter}.pth"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(policy.state_dict(), path)

        tqdm.write(f"iter {i_iter:4d}, avg_reward: {avg_reward:.2f}, {log_str}")

    print(f"used_time(s): {time.time() - start:.1f}")

    os.makedirs(os.path.dirname(cfg["save_path"]), exist_ok=True)
    torch.save(policy.state_dict(), cfg["save_path"])
    print(f"Model saved to {cfg['save_path']}")

    envs.close()

    plt.plot(range(len(return_list)), return_list)
    plt.xlabel("Iterations")
    plt.ylabel("Avg Return")
    plt.title(f"{algo.upper()} on {meta['env_name']}")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统一 RL 训练脚本。配置文件放在 config/ 目录下。"
    )
    parser.add_argument("--config", default="grpo_cartpole.json",
                        help="配置文件名（位于 config/ 目录下），默认 grpo_cartpole.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    algo = cfg.pop("algo")
    env_name = cfg.pop("env")
    train(algo, env_name, cfg)
