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
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from src.env import ENV_REGISTRY
from src.policy import POLICY_REGISTRY
from src.rl.grpo import GRPO
from src.rl.ppo import PPO
from src.policy.value_net import ValueNet


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
        path = config_file
    else:
        path = os.path.join(CONFIG_DIR, config_file)
    return load_json(path)


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------
def train(algo: str, env_name: str, cfg: dict):
    env_cls = ENV_REGISTRY[env_name]
    env = env_cls(num_envs=cfg["num_envs"])
    policy_cls = POLICY_REGISTRY[cfg["policy"]]

    print(f"Training {algo.upper()} on {env_cls.env_name}  [{cfg['policy']}]")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    max_steps = cfg["max_steps"]
    train_steps = cfg["train_steps"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = policy_cls(env.state_dim, env.action_dim).to(device)

    if algo == "grpo":
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
        algo_inst = GRPO(optimizer, eps=cfg["eps"],
                         n_iterations=cfg["n_iterations"],
                         discrete=env_cls.discrete,
                         entropy_coef=cfg.get("entropy_coef", 0.0))
    else:  # ppo
        critic = ValueNet(env.state_dim).to(device)
        actor_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["actor_lr"])
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg["critic_lr"])
        algo_inst = PPO(actor_optimizer, critic_optimizer,
                        eps=cfg["eps"], gamma=cfg["gamma"],
                        lmbda=cfg["lmbda"],
                        n_iterations=cfg["n_iterations"],
                        discrete=env_cls.discrete)

    return_list = []
    start = time.time()

    for i_iter in tqdm(range(train_steps)):
        if algo == "grpo":
            trajectories, episode_rewards = env.collect_trajectories(
                policy, max_steps, device=device)
            loss = algo_inst.update(policy, trajectories)
            log_str = f"loss: {loss:.4f}"
        else:  # ppo
            trajectories, episode_rewards = env.collect_step_data(
                policy, max_steps, device=device)
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

    env.close()

    os.makedirs("output", exist_ok=True)
    fig_path = f"output/{algo}_{env_name}_train.png"
    plt.plot(range(len(return_list)), return_list)
    plt.xlabel("Iterations")
    plt.ylabel("Avg Return")
    plt.title(f"{algo.upper()} on {env_cls.env_name}")
    plt.grid(True)
    plt.savefig(fig_path)
    plt.close()
    print(f"Figure saved to {fig_path}")


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
