import numpy as np
np.bool8 = np.bool_

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 区别 1: 输出层接 Softmax，输出动作的概率分布
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device
        # 区别 2: 不需要 ReplayBuffer，只需要存一局的数据
        self.log_probs = []  # 存储动作的对数概率
        self.rewards = []  # 存储即时奖励

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        # 创建分类分布
        action_dist = torch.distributions.Categorical(probs)
        # 根据概率采样一个动作
        action = action_dist.sample()
        # 存储该动作的 log_prob，用于后续算梯度
        self.log_probs.append(action_dist.log_prob(action))
        return action.item()

    def update(self):
        # 计算每一步的回报 G_t (从后往前算)
        G = 0
        self.optimizer.zero_grad()

        # 这里的 returns 列表存储的是每一时刻 t 对应的 G_t
        returns = []
        for r in self.rewards[::-1]:
            G = r + self.gamma * G
            returns.insert(0, G)

        # 标准化回报 (Optional, 但能极大地稳定训练)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        # Loss = - log(prob) * return
        for log_prob, G_t in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G_t)

        # 求和得到整局的 Loss
        loss = torch.stack(policy_loss).sum()

        loss.backward()
        self.optimizer.step()

        # 清空本局数据
        self.log_probs = []
        self.rewards = []


def run_REINFORCE():
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    done = done or truncated

                    # 存储 reward，但不需要存 state/next_state (因为是 MC 方法)
                    agent.rewards.append(reward)
                    state = next_state
                    episode_return += reward

                # 一局结束，立即更新
                agent.update()

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)


if __name__ == "__main__":
    run_REINFORCE()