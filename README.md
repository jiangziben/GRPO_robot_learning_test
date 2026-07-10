# GRPO vs PPO: 强化学习算法对比实验

本项目实现了 **GRPO（Group Relative Policy Optimization，组相对策略优化）** 与 **PPO（Proximal Policy Optimization，近端策略优化）** 两种强化学习算法，并在经典 Gym 控制环境中进行对比实验。

GRPO 是 DeepSeek-R1 中使用的强化学习算法，其核心思想是**无需 Critic 网络**，通过对同一组内多个 episode 的累积奖励进行组内归一化来计算优势函数，从而简化训练流程。

## 环境

| 环境 | 动作空间 | 状态维度 | 动作维度 |
|------|---------|---------|---------|
| [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | 离散 (Discrete) | 4 | 2 |
| [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/) | 连续 (Continuous) | 3 | 1 |

## 算法对比

| 特性 | PPO | GRPO |
|------|-----|------|
| 网络架构 | Actor-Critic（策略网络 + 价值网络） | 仅策略网络（无 Critic） |
| 优势估计 | GAE（广义优势估计） | 组内奖励归一化（Z-score） |
| 并行采样 | 单环境串行采样 | 多环境向量化并行采样 |
| 超参数数量 | 较多（需调节 Actor/Critic 学习率等） | 较少（仅需调节策略网络学习率） |

### GRPO 优势计算

$$\text{Advantage} = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

其中 $R_i$ 为组内第 $i$ 个 episode 的累积奖励，$\mu_R$ 和 $\sigma_R$ 分别为组内奖励的均值和标准差。

## 项目结构

```
.
├── rl_utils.py                  # 共享工具模块（训练循环、GAE、经验回放缓冲区）
├── grpo_train_cartpole.py       # GRPO 在 CartPole 上的训练脚本
├── grpo_train_pendulum.py       # GRPO 在 Pendulum 上的训练脚本
├── ppo_train_cartpole.py        # PPO 在 CartPole 上的训练脚本
├── ppo_train_pendulum.py        # PPO 在 Pendulum 上的训练脚本
├── grpo_test_cartpole.py        # GRPO CartPole 评估脚本
├── grpo_test_pendulum.py        # GRPO Pendulum 评估脚本
├── ppo_test_cartpole.py         # PPO CartPole 评估脚本
├── ppo_test_pendulum.py         # PPO Pendulum 评估脚本
├── doc/                         # 实验结果（GIF 动图 & 奖励曲线）
│   ├── result_grpo_wo_pos_reward.gif
│   ├── result_grpo_w_pos_reward.gif
│   ├── result_ppo_wo_pos_reward.gif
│   ├── return_grpo_wo_pos_reward_26.56s.png
│   ├── return_grpo_w_pos_reward_24.16s.png
│   └── return_ppo_wo_pos_reward_77.53s.png
└── weights/                     # 保存的模型权重（*.pth，已 gitignore）
```

## 环境配置

### 依赖

- Python 3.9+
- PyTorch 2.7+
- Gym 0.25.2
- NumPy 1.23.5（必须锁定版本，gym 0.25.2 不兼容 NumPy 2.x / 1.24+）
- Matplotlib
- tqdm
- Pygame（用于环境渲染）

### 安装

```bash
# 创建 conda 环境
conda create -n grpo-rl python=3.9 -y
conda activate grpo-rl

# 安装依赖
pip install -r requirements.txt
```

## 运行方式

### 训练

```bash
# GRPO 训练
python grpo_train_cartpole.py      # CartPole（离散动作）
python grpo_train_pendulum.py      # Pendulum（连续动作）

# PPO 训练
python ppo_train_cartpole.py       # CartPole（离散动作）
python ppo_train_pendulum.py       # Pendulum（连续动作）
```

训练完成后，模型权重将保存至 `./weights/` 目录。

### 评估

```bash
# GRPO 评估（需先完成训练）
python grpo_test_cartpole.py
python grpo_test_pendulum.py

# PPO 评估（需先完成训练）
python ppo_test_cartpole.py
python ppo_test_pendulum.py
```

评估脚本会加载训练好的模型，在环境中运行 10 个 episode 并渲染可视化，同时绘制每个 episode 的累积奖励曲线。

## 超参数

### GRPO

| 参数 | CartPole | Pendulum |
|------|----------|----------|
| group_size | 10 | 100 |
| 训练轮数 (episodes) | 50 | 500 |
| 最大步数 | 500 | 500 |
| 学习率 (lr) | 0.02 | 0.001 |
| PPO Clip (ε) | 0.2 | 0.2 |
| 更新迭代次数 | 20 | 20 |

### PPO

| 参数 | CartPole | Pendulum |
|------|----------|----------|
| 训练轮数 (episodes) | 500 | 5000 |
| Actor 学习率 | 1e-3 | 1e-4 |
| Critic 学习率 | 1e-2 | 5e-3 |
| Gamma (γ) | 0.98 | 0.9 |
| Lambda (λ) | 0.95 | 0.9 |
| PPO Clip (ε) | 0.2 | 0.2 |
| 更新轮数 (epochs) | 10 | 10 |

## 实验结果

以下为 CartPole-v1 环境下的训练结果对比：

### 奖励曲线

| GRPO（无位置奖励, 26.56s） | GRPO（有位置奖励, 24.16s） | PPO（无位置奖励, 77.53s） |
|:---:|:---:|:---:|
| ![GRPO without pos reward](doc/return_grpo_wo_pos_reward_26.56s.png) | ![GRPO with pos reward](doc/return_grpo_w_pos_reward_24.16s.png) | ![PPO without pos reward](doc/return_ppo_wo_pos_reward_77.53s.png) |

### 训练效果演示

| GRPO（无位置奖励） | GRPO（有位置奖励） | PPO（无位置奖励） |
|:---:|:---:|:---:|
| ![GRPO wo pos](doc/result_grpo_wo_pos_reward.gif) | ![GRPO w pos](doc/result_grpo_w_pos_reward.gif) | ![PPO wo pos](doc/result_ppo_wo_pos_reward.gif) |

> **位置奖励**：在 CartPole 环境中额外添加了 `-abs(cart_position)` 惩罚项，鼓励小车保持在轨道中心附近。

## 参考资料

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)
