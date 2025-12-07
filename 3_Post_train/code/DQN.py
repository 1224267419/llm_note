import random
import numpy as np
# numpy版本问题,补充np.bool8 = np.bool_
np.bool8 = np.bool_
import gym
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class ReplayBuffer:
    "经验回放池"
    def __init__(self, capacity):
        # FIFO的队列,
        # maxlen=capacity:如果存满了，新数据进来会自动把最旧的数据挤出去
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

class ConvolutionalQnet(torch.nn.Module):
    '''
    加入卷积层的Q网络 在一些视频游戏中，智能体并不能直接获取这些状态信息，而只能直接获取屏幕中的图像。
    要让智能体和人一样玩游戏，我们需要让智能体学会以图像作为状态时的决策。
    我们可以利用 7.4 节的 DQN 算法，将卷积层加入其网络结构以提取图像特征，
    最终实现以图像为输入的强化学习
    '''

    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)

class DQN:
    "定义DQN算法"
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 优化器只更新 q_net 的参数,target_q_net的参数不需要每轮更新
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        # target-network更新频率
        self.target_update = target_update
        self.count = 0
        self.device=device
    def take_action(self, state):
        """
        epsilon-greedy算法
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    def update(self, transition_dict):
        # 把数据都转成 Tensor 放到 GPU/CPU 上
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # .gather(dim, index):在指定维度（dim）上，根据索引（index）取出对应位置的元素
        # action告诉我们我们采取哪个动作,self.q_net输出策略在对应位置的Q 值
        # .gather(1, actions) 我们采取的那个动作对应的 Q 值
        q_values = self.q_net(states).gather(1, actions)
        # Q-learning 用target_net算出下一步状态 next_states 的所有 Q 值，取最大
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # Target = Reward + Gamma * Max(Next_Q) * (1 - Done),
        # 如果 Done 了（游戏结束），就没有下一步价值了 ,所以乘以 (1 - dones)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 均方差,用mean处理一个batch
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        # 每隔 target_update 步，把 q_net 的参数完全复制给 target_q_net
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) # 更新target-network
        self.count+=1

class DoubleDQN(DQN):
    "继承DQN算法"

    def update(self, transition_dict):
        # 把数据都转成 Tensor 放到 GPU/CPU 上
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # 2. 计算当前 Q 值 (和 DQN 一样)
        q_values = self.q_net(states).gather(1, actions)

        # ================== Double DQN 的核心修改点 ==================

        # 步骤 A: 使用【当前网络 (q_net)】计算下一状态的最佳动作索引
        # .max(1)[1] 返回的是索引 (argmax action)，而不是数值
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)

        # 步骤 B: 使用【目标网络 (target_q_net)】计算该动作对应的 Q 值
        # .gather(1, max_action) 根据上面选出的动作索引提取 Q 值
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 均方差,用mean处理一个batch
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        # 每隔 target_update 步，把 q_net 的参数完全复制给 target_q_net
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) # 更新target-network
        self.count+=1

def run_DQN(network):
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda")

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    # env.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = network(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                state = state[0]  # 因为新版 Gym 返回 (data, info)，我们要取第0个数据
                done = False
                # 一个 Episode
                while not done:
                    action = agent.take_action(state)
                    # # 新版 Gym 返回 5 个值：next_state, reward, terminated, truncated, info
                    next_state, reward, done, truncated, _ = env.step(action)
                    # 环境自然结束（Terminated，包括目标成功，失败等）
                    # 和人为截断（Truncated，主要为达到一定步数结束）
                    done = done or truncated
                    # q-learn更新需要的4part+done
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        # 从池子里随机抽 batch_size (64) 条数据
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                # 记录这一局的总得分
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
if __name__ == "__main__":
    # run_DQN(DQN)
    run_DQN(DoubleDQN)