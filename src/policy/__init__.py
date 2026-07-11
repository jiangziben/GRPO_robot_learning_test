"""模型模块。"""

from .carpole_policy import PolicyNet
from .pendulum_policy import PolicyNetContinuous

POLICY_REGISTRY = {
    "PolicyNet": PolicyNet,
    "PolicyNetContinuous": PolicyNetContinuous,
}
