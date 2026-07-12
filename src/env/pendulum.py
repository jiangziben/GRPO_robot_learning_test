"""Pendulum-v1 环境。"""

from .base import BaseEnv


class PendulumEnv(BaseEnv):
    env_name = "Pendulum-v1"
    discrete = False

    def reward(self, next_states, rewards, dones):
        return (rewards + 8.0) / 8.0

    @staticmethod
    def is_success(state, steps: int, done: bool) -> bool:
        # Pendulum state = [cos(θ), sin(θ), θ_dot]
        # cos(θ) ≈ 1 表示终止时接近直立
        return state[0] > 0.95
