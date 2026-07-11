"""GRPO 统一测试脚本。支持 CartPole 和 Pendulum 环境。

用法:
    python grpo_test.py --env cartpole --model weights/grpo_cartpole_policy_final.pth
    python grpo_test.py --env pendulum --model weights/grpo_pendulum_policy_final.pth
"""

import argparse
import gym
import torch
import matplotlib.pyplot as plt
from model import PolicyNet, PolicyNetContinuous


ENV_CONFIGS = {
    "cartpole": {
        "env_name": "CartPole-v1",
        "policy_cls": PolicyNet,
    },
    "pendulum": {
        "env_name": "Pendulum-v1",
        "policy_cls": PolicyNetContinuous,
    },
}


def test(env_name: str, model_path: str, num_episodes: int = 10):
    cfg = ENV_CONFIGS[env_name]

    env = gym.make(cfg["env_name"], render_mode="human")
    state_dim = env.observation_space.shape[0]
    n_actions = (env.action_space.n if env_name == "cartpole"
                 else env.action_space.shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = cfg["policy_cls"](state_dim, n_actions).to(device)
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
    plt.title(f"Test Rewards on {cfg['env_name']}")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "pendulum"], required=True)
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    test(args.env, args.model, args.episodes)
