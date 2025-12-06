import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class ReplayBuffer:
    "经验回放池"
    def __init__(self, capacity):
        # FIFO的队列
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # 加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 采样,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        # 假设transitions = [(s0,a0,r0,s0',False), (s1,a1,r1,s1',True)]（随机采样，这里假设顺序不变）；
        # zip(*transitions) 后得到：
        # state = (s0, s1)，action = (a0, a1)，reward = (r0, r1)，next_state = (s0', s1')，done = (False, True)；
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        # 目前buffer中数据的数量
        return len(self.buffer)

class Qnet(torch.nn.Module):
    "定义一个简单的一层网络"
    def __init__(self,state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
class DQN:
    "定义DQN算法"
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        # target-network更新频率
        self.target_update = target_update
        self.count = 0
        self.device=device
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    def update(self, transition_dict):