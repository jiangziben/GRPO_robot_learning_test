# 常用库
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
# PyTorch相关
import torch
from torch.nn import functional as F
import gym
import numpy as np
    
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc_mu = torch.nn.Linear(128, action_dim)
        self.fc_std = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
def take_action(state,policy_net, device):
        state = torch.tensor([state], dtype=torch.float).to(device)
        mu, sigma = policy_net(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]


def collect_trajectory_vectorized(envs, policy_net, num_steps=200, gamma=0.99, device="cpu"):
    """
    从 vectorized 环境并行采样多个轨迹，计算归一化奖励
    envs: 并行环境 (vectorized environment)
    policy_net: 策略网络 (PolicyNet)
    num_steps: 每条轨迹最大步数
    gamma: 折扣因子

    返回：
        states, log_probs, actions, normalized_rewards
    """
    group_size = envs.num_envs  # 获取并行环境数量
    seed_num = np.random.randint(0, 1000)
    states = envs.reset(seed = [seed_num] * group_size)  # shape: [group_size, state_dim]

    all_states = []
    all_actions = []
    all_log_probs = []
    all_rewards = torch.zeros(group_size)  # shape: [group_size]
    for t in range(num_steps):
        states_tensor = torch.tensor(states, dtype=torch.float32,device=device)  # shape: [group_size, state_dim]
        mu, sigma = policy_net(states_tensor)
        action_dist = torch.distributions.Normal(mu, sigma)
        actions = action_dist.sample()
        log_probs =  action_dist.log_prob(actions).detach()

        # 执行环境步进
        next_states, rewards, dones, infos = envs.step(actions.cpu().numpy())

        all_states.append(states)
        all_actions.append(actions)
        all_log_probs.append(log_probs)
        rewards += next_states[:,0]
        all_rewards += rewards   # shape: [group_size]

        states = next_states
        if np.all(dones):  # 如果所有环境都结束，则停止t
            break
    normalized_rewards = (all_rewards / num_steps).to(device)
    all_states = torch.tensor(np.array(all_states)).permute(1,0,2).to(device)
    all_log_probs = torch.stack(all_log_probs).permute(1,0,2).to(device)
    all_actions = torch.stack(all_actions).permute(1,0,2).to(device)
    trajectories = {"all_states":all_states,"all_log_probs": all_log_probs,
                    "all_actions": all_actions,"normalized_rewards": normalized_rewards}
    episode_rewards = (normalized_rewards * num_steps)
    return trajectories,episode_rewards

def calc_advantages_with_grpo(trajectories):
    """从轨迹中提取奖励，并标准化每个episode的奖励"""
    rewards = trajectories["normalized_rewards"]  # 提取最终奖励，
    mean_reward = torch.mean(rewards)     # 计算平均值，
    std_reward = torch.std(rewards)  + 1e-8          # 计算标准差（1e-8是防止0除），
    advantages = (rewards - mean_reward)/std_reward  # 最后标准化每个episode

    return advantages

def grpo_update(trajectories, net, optimizer, n_iterations=20, eps=0.2):

    # [1] 使用GRPO函数计算每个episode的标准化Advantage
    advantages = calc_advantages_with_grpo(trajectories).unsqueeze(-1)  # [batch_size, 1]
    # 将所有轨迹的数据合并成批处理
    all_states = trajectories["all_states"]          # [batch_size, num_steps, state_dim]
    all_log_probs = trajectories["all_log_probs"]    # [batch_size, num_steps, 1]
    all_chosen_actions = trajectories["all_actions"] # [batch_size, num_steps, 1]

    # [2] 更新Policy NN。进行n_iterations次更新
    for i_iter in range(n_iterations):
        mu, sigma = net(all_states)  # [B, T, 1]，nn.Linear自动处理3D输入
        action_dist = torch.distributions.Normal(mu, sigma)
        new_log_probs = action_dist.log_prob(all_chosen_actions)  # [B, T, 1]
        ratio = torch.exp(new_log_probs - all_log_probs)          # [B, T, 1]
        surr1 = ratio * advantages.unsqueeze(1)  # advantages: [B, 1] → [B, 1, 1]
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages.unsqueeze(1)  # 截断
        loss = torch.mean(-torch.min(surr1, surr2))

        # [3] 更新Policy NN的权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

# [1] 初始化和初始设置
group_size = 100
env_name = 'Pendulum-v1'
envs = gym.vector.make(env_name,num_envs=group_size)
state_dim = envs.single_observation_space.shape[0]  # 3
n_actions = envs.single_action_space.shape[0]  # 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
policy = PolicyNetContinuous(state_dim, n_actions).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
max_steps = 500
# [3] 进行500次iteration循环
iteration_num = 500  
start = time.time()
count = 0
return_list = []
# [4] 开始500次iteration循环
for iter in tqdm(range(iteration_num)):  
    # [5] 使用GRPO积累轨迹
    trajectories,episode_rewards = collect_trajectory_vectorized(envs,policy,max_steps,device=device)

    # [6] 使用GRPO更新PolicyNet的权重
    loss = grpo_update(trajectories, policy, optimizer)
    
    # [7] 计算平均奖励
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    return_list.append(avg_reward.cpu().numpy())
    if iter !=0 and iter % 200 == 0:
        save_path = f"./weights/grpo_pendulum_policy_update_{iter}.pth"
        torch.save(policy.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    print(f'第 {iter} 次试验, avg reward: {avg_reward:.2f}')    

print("used_time(s): ", time.time() - start)

save_path = f"./weights/grpo_pendulum_policy_update_final.pth"
torch.save(policy.state_dict(), save_path)
print(f"Model saved to {save_path}")

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('GRPO on {}'.format(env_name))
plt.grid(True)
plt.show()
envs.close()
