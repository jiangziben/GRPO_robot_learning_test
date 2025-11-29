# https://cloud.tencent.com/developer/article/2387660



import gym
import matplotlib.pyplot as plt
import torch
from ppo_test_pendulum import *

# 创建环境和模型
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("Pendulum-v1", render_mode="human")

class config:
    # env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]  # 4
    n_actions = env.action_space.shape[0]       # 2
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
agent = PPOContinuous(config.state_dim, config.hidden_dim, config.n_actions, config.actor_lr, config.critic_lr, config.lmbda,
        config.epochs, config.eps, config.gamma, config.device)
model_path = "./weights/ppo_pendulum_policy_update_final.pth"
model = torch.load(model_path, map_location = 'cpu')
agent.actor.load_state_dict(model)
agent.actor.eval()  # 设置为评估模式

model = agent


# 训练模型
episodes = 100
rewards = []

for episode in range(episodes):
    observation = env.reset()
    total_reward = 0

    while True:
        # 模型根据观测选择动作
        action = model.take_action(observation)

        # 在环境中执行动作
        next_observation, reward, done, _ = env.step(action)

        # 更新总奖励
        total_reward += reward

        # 可视化环境状态
        env.render()

        # 更新观测
        observation = next_observation

        # 判断是否结束
        if done:
            break

    rewards.append(total_reward)

# 可视化训练过程中的奖励变化
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Process")
plt.show()

# 关闭环境渲染
env.close()