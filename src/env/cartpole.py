"""CartPole-v1 环境。"""

import numpy as np

from .base import BaseEnv


class CartPoleEnv(BaseEnv):
    env_name = "CartPole-v1"
    discrete = True

    def reward(self, next_states, rewards, dones):
        rewards = rewards.copy()
        rewards += -np.abs(next_states[:, 0])
        return rewards

    @staticmethod
    def is_success(state, steps: int, done: bool) -> bool:
        # CartPole-v1 默认 max_episode_steps=500，跑满即成功
        return steps >= 500
