# https://cloud.tencent.com/developer/article/2387660

import gym
import shap
import numpy as np
import torch

# 创建环境和模型
env = gym.make("CartPole-v1")
model_path = "./weights/ppo_cartpole_policy_update_final.pth"
model = torch.load(model_path, map_location = 'cpu')
# model = YourModel()  # 替换成你的强化学习模型

# 创建解释器
explainer = shap.Explainer(model, env.observation_space.sample())

# 解释一个样本
sample_observation = env.observation_space.sample()
shap_values = explainer.shap_values(sample_observation)

# 可视化解释结果
shap.summary_plot(shap_values, sample_observation)