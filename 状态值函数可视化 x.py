# https://cloud.tencent.com/developer/article/2387660

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from ppo_test_cartpole import *

# 创建环境和模型
env = gym.make("CartPole-v1")

class config:
    # env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]  # 4
    n_actions = env.action_space.n              # 2
    actor_lr = 1e-3
    critic_lr = 1e-2
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cpu")

torch.manual_seed(0)
# 初始化策略网络并加载模型参数
agent = PPO(config.state_dim, config.hidden_dim, config.n_actions, config.actor_lr, config.critic_lr, config.lmbda,
        config.epochs, config.eps, config.gamma, config.device)
model_path = "./weights/ppo_cartpole_policy_update_final.pth"
model = torch.load(model_path, map_location = 'cpu')
agent.actor.load_state_dict(model)
agent.actor.eval()  # 设置为评估模式

model = agent.actor

# 计算状态值函数
states = np.linspace(env.observation_space.low, env.observation_space.high, num=100)
values = np.zeros_like(states[:, 0])

for i, state in enumerate(states):
    values[i] = model.calculate_state_value(state)

# 可视化状态值函数
plt.plot(states[:, 0], values)
plt.xlabel("Position")
plt.ylabel("State Value")
plt.title("State Value Function")
plt.show()