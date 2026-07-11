"""策略模块。"""

from .discrete_policy import PolicyNet
from .continuous_policy import PolicyNetContinuous

POLICY_REGISTRY = {
    "PolicyNet": PolicyNet,
    "PolicyNetContinuous": PolicyNetContinuous,
}
