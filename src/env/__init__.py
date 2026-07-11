"""环境模块。每个环境类封装 gym env、reward 计算和轨迹收集。"""

from .cartpole import CartPoleEnv
from .pendulum import PendulumEnv

ENV_REGISTRY = {
    "cartpole": CartPoleEnv,
    "pendulum": PendulumEnv,
}
