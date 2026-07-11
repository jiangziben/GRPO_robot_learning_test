"""统一测试脚本。支持 GRPO 和 PPO，支持 CartPole 和 Pendulum 环境。

配置文件（含 algo、env）放在 config/ 目录下。

用法:
    python test.py --config grpo_cartpole.json --model weights/grpo_cartpole_policy_final.pth
    python test.py --config ppo_pendulum.json --model weights/ppo_pendulum_policy_final.pth
"""

import argparse
import json
import os
import gym
import torch
import matplotlib.pyplot as plt

from src.model.carpole_policy import PolicyNet
from src.model.pendulum_policy import PolicyNetContinuous


# ---------------------------------------------------------------------------
# 环境元信息
# ---------------------------------------------------------------------------
ENV_META = {
    "cartpole": {
        "env_name": "CartPole-v1",
        "policy_cls": PolicyNet,
        "discrete": True,
    },
    "pendulum": {
        "env_name": "Pendulum-v1",
        "policy_cls": PolicyNetContinuous,
        "discrete": False,
    },
}

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------
def test(env_name: str, model_path: str, num_episodes: int = 10):
    meta = ENV_META[env_name]

    env = gym.make(meta["env_name"], render_mode="human")
    state_dim = env.observation_space.shape[0]
    n_actions = (env.action_space.n if meta["discrete"]
                 else env.action_space.shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = meta["policy_cls"](state_dim, n_actions).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    episode_rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = policy.take_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}, Steps = {steps}")
    env.close()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_episodes + 1), episode_rewards, marker="o", linestyle="-")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Test Rewards on {meta['env_name']}")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统一 RL 测试脚本。配置文件放在 config/ 目录下。"
    )
    parser.add_argument("--config", default="grpo_cartpole.json",
                        help="配置文件名（位于 config/ 目录下），默认 grpo_cartpole.json")
    parser.add_argument("--model", default=None,
                        help="模型权重路径，默认使用配置文件中的 save_path")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    config_path = args.config
    if os.path.sep not in config_path and os.path.altsep not in config_path:
        config_path = os.path.join(CONFIG_DIR, config_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_path = args.model or cfg["save_path"]
    test(cfg["env"], model_path, args.episodes)
