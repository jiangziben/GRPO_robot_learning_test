"""Pendulum-v1 环境。"""

from .base import BaseEnv


class PendulumEnv(BaseEnv):
    env_name = "Pendulum-v1"
    discrete = False

    def reward(self, next_states, rewards, dones):
        return (rewards + 8.0) / 8.0
