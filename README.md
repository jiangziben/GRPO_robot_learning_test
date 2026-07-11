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
| 并行采样 | 向量化并行采样 | 向量化并行采样 |
| 超参数 | 较多（Actor/Critic 学习率、γ、λ） | 较少（策略学习率、熵系数） |
| 适用场景 | 通用（离散/连续动作均稳定） | 离散动作效果好，连续动作需熵正则 |

### GRPO 优势计算

$$\text{Advantage} = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

其中 $R_i$ 为组内第 $i$ 个 episode 的累积奖励，$\mu_R$ 和 $\sigma_R$ 分别为组内奖励的均值和标准差。

## 项目结构

```
.
├── train.py                      # 统一训练脚本
├── test.py                       # 统一测试脚本
├── config/                       # 配置文件（超参、algo、env）
│   ├── grpo_cartpole.json
│   ├── grpo_pendulum.json
│   ├── ppo_cartpole.json
│   └── ppo_pendulum.json
├── src/
│   ├── model/                    # 策略/价值网络
│   │   ├── carpole_policy.py     # PolicyNet（离散）
│   │   ├── pendulum_policy.py    # PolicyNetContinuous（连续）
│   │   └── value_net.py          # ValueNet（PPO Critic）
│   ├── rl/                       # 算法实现
│   │   ├── grpo.py               # GRPO 类
│   │   └── ppo.py                # PPO 类 + GAE
│   └── utils/
│       └── utils.py              # 轨迹收集工具函数
├── output/                       # 训练/测试结果图（自动生成）
├── weights/                      # 模型权重（gitignore）
└── doc/                          # 实验素材
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
# 使用配置文件运行训练（algo 和 env 在配置文件中指定）
python train.py --config grpo_cartpole.json
python train.py --config grpo_pendulum.json
python train.py --config ppo_cartpole.json
python train.py --config ppo_pendulum.json

# 不传 --config 默认使用 grpo_cartpole.json
python train.py
```

训练完成后的输出：
- 模型权重：`weights/{algo}_{env}_policy_final.pth`
- 奖励曲线：`output/{algo}_{env}_train.png`

### 测试

```bash
# 使用配置文件运行测试（--model 可选，默认用配置文件中的 save_path）
python test.py --config grpo_cartpole.json
python test.py --config ppo_pendulum.json --model weights/my_model.pth

# 不传 --config 默认使用 grpo_cartpole.json
python test.py --model weights/grpo_cartpole_policy_final.pth
```

测试完成后的输出：
- 测试奖励曲线：`output/{env}_test.png`

### 配置文件格式

每个配置文件自包含 algo、env 及所有超参，示例（`config/grpo_cartpole.json`）：

```json
{
    "algo": "grpo",
    "env": "cartpole",
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
| entropy_coef | 0（不需要） | 0.01（防 sigma 坍缩） |

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

- **GRPO + Pendulum**：由于 Pendulum 是长时域（200 步）连续控制任务，GRPO 仅用最终 episode reward 做组内比较，缺少 per-step 信用分配，训练存在随机种子敏感性。加入熵正则（`entropy_coef=0.01`）可缓解策略坍缩，但稳定性仍不如 PPO + GAE。
- **GRPO + CartPole**：训练稳定，收敛快，是 GRPO 的理想适用场景。

## 参考资料

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)
