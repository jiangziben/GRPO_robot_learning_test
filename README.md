# GRPO & PPO 强化学习算法对比实验

本项目实现了 **GRPO（Group Relative Policy Optimization）** 与 **PPO（Proximal Policy Optimization）** 两种强化学习算法，在经典 Gym 控制环境中进行对比实验。

GRPO 是 DeepSeek-R1 中使用的强化学习算法，核心思想是**无需 Critic 网络**，通过对同一组内多个 episode 的累积奖励进行组内归一化来计算优势函数，从而简化训练流程。

## 环境

| 环境 | 动作空间 | 状态维度 | 动作维度 |
|------|---------|---------|---------|
| [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | 离散 (Discrete) | 4 | 2 |
| [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/) | 连续 (Continuous) | 3 | 1 |

## 算法对比

| 特性 | PPO | GRPO |
|------|-----|------|
| 网络架构 | Actor-Critic（策略网络 + 价值网络） | 仅策略网络（无 Critic） |
| 优势估计 | GAE（广义优势估计，per-step） | 组内奖励归一化（Z-score，per-episode） |
| 数据收集 | step 级（保留每步 reward/done/next_state） | 轨迹级（仅保留累计 reward） |
| 超参数 | 较多（Actor/Critic 学习率、γ、λ） | 较少（策略学习率、熵系数） |
| 适用场景 | 通用（离散/连续动作均稳定） | 离散动作效果好，连续动作需熵正则 |

### GRPO 优势计算

$$\text{Advantage} = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

其中 $R_i$ 为组内第 $i$ 个 episode 的累积奖励，$\mu_R$ 和 $\sigma_R$ 分别为组内奖励的均值和标准差。

## 项目结构

```
.
├── train.py                       # 统一训练脚本
├── test.py                        # 统一测试脚本
├── config/                        # 配置文件（algo、env、policy 及超参）
│   ├── grpo_cartpole.json
│   ├── grpo_pendulum.json
│   ├── ppo_cartpole.json
│   └── ppo_pendulum.json
├── src/
│   ├── env/                       # 环境封装（gym env + reward + 轨迹收集）
│   │   ├── base.py                #   BaseEnv：collect_trajectories / collect_step_data
│   │   ├── cartpole.py            #   CartPoleEnv：reward = 原始 reward + 中心惩罚
│   │   └── pendulum.py            #   PendulumEnv：reward = (r + 8) / 8
│   ├── policy/                    # 策略/价值网络
│   │   ├── discrete_policy.py     #   PolicyNet（离散）
│   │   ├── continuous_policy.py   #   PolicyNetContinuous（连续）
│   │   └── value_net.py           #   ValueNet（PPO Critic）
│   └── rl/                        # 强化学习算法
│       ├── grpo.py                #   GRPO：组内标准化 + PPO clip + 熵正则
│       └── ppo.py                 #   PPO：GAE + Actor-Critic
├── output/                        # 训练/测试奖励曲线（自动生成）
├── weights/                       # 模型权重（gitignore）
└── doc/                           # 实验素材
```

## 架构设计

### 环境层 (`src/env/`)

每个环境类封装了与该环境相关的全部逻辑，**不需要在 train.py 中写任何环境特定代码**：

| 职责 | 来源 |
|------|------|
| gym env 创建 | `BaseEnv.__init__` |
| 动作空间类型 (discrete) | 类属性 |
| reward 变换 | `env.reward()` |
| 轨迹级数据收集（GRPO） | `env.collect_trajectories(policy, num_steps, device)` |
| Step 级数据收集（PPO） | `env.collect_step_data(policy, num_steps, device)` |

添加新环境只需创建 `src/env/xxx.py` 并注册到 `ENV_REGISTRY`。

### 策略层 (`src/policy/`)

策略网络通过 `POLICY_REGISTRY` 按名称查找，用户在配置文件中指定：

```json
{ "policy": "PolicyNet" }           // CartPole
{ "policy": "PolicyNetContinuous" } // Pendulum
```

每个策略提供两个接口：
- `sample_with_log_prob(state_tensor)` — 训练时批处理采样，返回 `(actions, log_probs)`
- `take_action(state)` — 推理时单步采样

### 数据流

```
config.json  →  train.py  →  ENV_REGISTRY[env]  →  env.collect_*()
                                POLICY_REGISTRY[policy]
                                GRPO / PPO
```

## 环境配置

### 依赖

- Python 3.9+
- PyTorch 2.0+
- Gym 0.25.2
- NumPy 1.23.5（必须锁定版本，gym 0.25.2 不兼容 NumPy 2.x / 1.24+）
- Matplotlib
- tqdm
- Pygame（用于环境渲染）

### 安装

```bash
conda create -n grpo-rl python=3.9 -y
conda activate grpo-rl
pip install -r requirements.txt
```

## 运行方式

### 训练

```bash
python train.py --config grpo_cartpole.json
python train.py --config grpo_pendulum.json
python train.py --config ppo_cartpole.json
python train.py --config ppo_pendulum.json

# 不传 --config 默认使用 grpo_cartpole.json
python train.py
```

输出：
- 模型权重：`weights/{algo}_{env}_policy_final.pth`
- 奖励曲线：`output/{algo}_{env}_train.png`

### 测试

```bash
# --model 可选，默认用配置文件中的 save_path
python test.py --config grpo_cartpole.json
python test.py --config ppo_pendulum.json --model weights/my_model.pth
```

输出：
- 测试奖励曲线：`output/{env}_test.png`

### 配置文件格式

```json
{
    "algo": "grpo",
    "env": "cartpole",
    "policy": "PolicyNet",
    "save_path": "weights/grpo_cartpole_policy_final.pth",
    "num_envs": 20,
    "iteration_num": 100,
    "max_steps": 500,
    "lr": 0.02,
    "n_iterations": 20,
    "eps": 0.2
}
```

## 超参数

### GRPO

| 参数 | CartPole | Pendulum |
|------|----------|----------|
| num_envs | 20 | 100 |
| iteration_num | 100 | 500 |
| max_steps | 500 | 500 |
| lr | 0.02 | 0.001 |
| n_iterations | 20 | 20 |
| eps | 0.2 | 0.2 |
| entropy_coef | — | 0.01（防 sigma 坍缩） |

### PPO

| 参数 | CartPole | Pendulum |
|------|----------|----------|
| num_envs | 20 | 20 |
| iteration_num | 200 | 500 |
| max_steps | 500 | 500 |
| actor_lr | 0.001 | 0.0001 |
| critic_lr | 0.01 | 0.005 |
| n_iterations | 10 | 10 |
| eps | 0.2 | 0.2 |
| gamma | 0.98 | 0.9 |
| lmbda | 0.95 | 0.9 |

## 已知局限

- **GRPO + Pendulum**：Pendulum 是长时域（200 步）连续控制任务，GRPO 仅用最终 episode reward 做组内比较，缺少 per-step 信用分配，训练存在随机种子敏感性。加入熵正则（`entropy_coef=0.01`）可缓解策略坍缩，但稳定性仍不如 PPO + GAE。
- **GRPO + CartPole**：训练稳定，收敛快，是 GRPO 的理想适用场景。

## 参考资料

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)
