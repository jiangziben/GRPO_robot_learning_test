"""环境基类。封装 gym vectorized env、reward 变换和轨迹收集。"""

import numpy as np
import torch
import gym


class BaseEnv:
    """环境基类。

    子类需定义：
        env_name:   gym 环境名
        discrete:   True=离散动作，False=连续动作

    子类需实现：
        reward(self, next_states, rewards, dones) -> rewards
    """

    env_name: str
    discrete: bool

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = gym.vector.make(self.env_name, num_envs=self.num_envs)

    @property
    def state_dim(self) -> int:
        return self.envs.single_observation_space.shape[0]

    @property
    def action_dim(self) -> int:
        if self.discrete:
            return self.envs.single_action_space.n
        return self.envs.single_action_space.shape[0]

    def reward(self, next_states, rewards, dones):
        """每步 reward 变换。子类实现。

        Args:
            next_states: [B, state_dim] 转移后的状态
            rewards:     [B] 原始 reward
            dones:       [B] 当前步是否终止（仅标记 episode 边界，不用于清零）
        Returns:
            rewards:     [B] 变换后的 reward
        """
        raise NotImplementedError

    @staticmethod
    def is_success(state, steps: int, done: bool) -> bool:
        """判断单个 episode 是否成功。子类可重写。

        Args:
            state: episode 结束时的状态（gym step 返回的 observation）
            steps: episode 运行的步数
            done:  是否因环境终止条件而结束

        Returns:
            是否判定为成功
        """
        return False

    def close(self):
        self.envs.close()

    # ------------------------------------------------------------------
    # 轨迹级收集（GRPO：只返回累计 reward，不保留每步细节）
    # ------------------------------------------------------------------
    def collect_trajectories(self, policy, num_steps=500, device="cpu"):
        """收集轨迹级数据（GRPO 使用）。

        Returns:
            trajectories:    {"all_states", "all_log_probs", "all_actions",
                              "episode_rewards", "all_masks"}
            episode_rewards: [B] 每个 env 的累计原始奖励（仅有效步，由 mask 保证）
        """
        B = self.num_envs
        seed_num = np.random.randint(0, 1000)
        states = self.envs.reset(seed=[seed_num] * B)

        all_states, all_actions, all_log_probs = [], [], []
        all_rewards = torch.zeros(B)
        all_dones = torch.tensor([False] * B)
        all_masks_list = []

        for _ in range(num_steps):
            states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
            actions, log_probs = policy.sample_with_log_prob(states_tensor)

            next_states, rewards, dones, infos = self.envs.step(actions.cpu().numpy())

            all_states.append(states)
            all_actions.append(actions)
            all_log_probs.append(log_probs)

            # 记录哪些 env 在当前步是有效的（尚未终止）
            active_mask = (~all_dones).clone().numpy()
            all_masks_list.append(torch.tensor(active_mask, dtype=torch.float32))

            all_dones[dones] = True
            rewards = self.reward(next_states, rewards, dones)
            # 仅将有效步的 reward 累加到 episode 总和中
            all_rewards += torch.tensor(rewards, dtype=torch.float32) * all_masks_list[-1]

            states = next_states
            if torch.all(all_dones):
                break

        all_masks = torch.stack(all_masks_list).permute(1, 0)           # [B, T]

        all_states = self._pack_states(all_states, device)
        all_log_probs = self._pack_log_probs(all_log_probs, device)
        all_actions = self._pack_actions(all_actions, device)
        all_masks = all_masks.to(device)

        # episode_rewards = mask 门控下的原始奖励累加（僵尸步已被 mask 排除）
        episode_rewards = all_rewards.to(device)

        trajectories = {
            "all_states": all_states,
            "all_log_probs": all_log_probs,
            "all_actions": all_actions,
            "episode_rewards": episode_rewards,
            "all_masks": all_masks,
        }
        return trajectories, episode_rewards

    # ------------------------------------------------------------------
    # Step 级收集（PPO：保留每步 reward / done / next_state，用于 GAE）
    # ------------------------------------------------------------------
    def collect_step_data(self, policy, num_steps=500, device="cpu"):
        """收集 step 级数据（PPO 使用）。

        Returns:
            trajectories:    {"all_states", "all_next_states", "all_log_probs",
                              "all_actions", "all_rewards", "all_dones", "all_masks"}
            episode_rewards: [B] 每个 env 的累计奖励（仅有效步）
        """
        B = self.num_envs
        seed_num = np.random.randint(0, 1000)
        states = self.envs.reset(seed=[seed_num] * B)

        all_states, all_next_states, all_actions, all_log_probs = [], [], [], []
        all_rewards, all_dones, all_masks = [], [], []
        all_episode_rewards = torch.zeros(B)
        all_episode_dones = torch.tensor([False] * B)

        for _ in range(num_steps):
            states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
            actions, log_probs = policy.sample_with_log_prob(states_tensor)

            next_states, rewards, dones, infos = self.envs.step(actions.cpu().numpy())

            all_states.append(states)
            all_next_states.append(next_states)
            all_actions.append(actions)
            all_log_probs.append(log_probs)

            # 记录哪些 env 在当前步是有效的（尚未终止）
            active_mask = (~all_episode_dones).clone().numpy()
            mask_tensor = torch.tensor(active_mask, dtype=torch.float32)
            all_masks.append(mask_tensor)

            all_episode_dones[dones] = True
            rewards = self.reward(next_states, rewards, dones)
            all_rewards.append(torch.tensor(rewards, dtype=torch.float32))
            all_dones.append(torch.tensor(dones, dtype=torch.float32))
            all_episode_rewards += torch.tensor(rewards, dtype=torch.float32) * mask_tensor

            states = next_states
            if torch.all(all_episode_dones):
                break

        T = len(all_states)
        all_states = self._pack_states(all_states, device)
        all_next_states = torch.tensor(np.array(all_next_states), dtype=torch.float32,
                                       device=device).permute(1, 0, 2)
        all_rewards = torch.stack(all_rewards, dim=0).permute(1, 0).to(device)
        all_dones = torch.stack(all_dones, dim=0).permute(1, 0).to(device)
        all_masks = torch.stack(all_masks, dim=0).permute(1, 0).to(device)
        all_log_probs = self._pack_log_probs(all_log_probs, device)
        all_actions = self._pack_actions(all_actions, device)

        trajectories = {
            "all_states": all_states,
            "all_next_states": all_next_states,
            "all_log_probs": all_log_probs,
            "all_actions": all_actions,
            "all_rewards": all_rewards,
            "all_dones": all_dones,
            "all_masks": all_masks,
        }
        episode_rewards = all_episode_rewards
        return trajectories, episode_rewards

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _pack_states(self, states_list, device):
        return torch.tensor(np.array(states_list), dtype=torch.float32,
                            device=device).permute(1, 0, 2)

    def _pack_log_probs(self, log_probs_list, device):
        """stack + permute → [B, T] (discrete) or [B, T, act_dim] (continuous)."""
        if self.discrete:
            return torch.stack(log_probs_list).permute(1, 0).to(device)
        return torch.stack(log_probs_list).permute(1, 0, 2).to(device)

    def _pack_actions(self, actions_list, device):
        if self.discrete:
            return torch.stack(actions_list).permute(1, 0).to(device)
        return torch.stack(actions_list).permute(1, 0, 2).to(device)
