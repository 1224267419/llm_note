import gym
import torch
import torch.nn.functional as F
import numpy as np
np.bool8 = np.bool_
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
    # 策略网络
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    # 价值评估
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO:
    '''截断方式的PPO '''
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr,critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
    def take_action(self, state):
        # policy-base的标准代码
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        # 上面的代码和actor_Critic部分的一致

        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta).cuda()
        # 我们通过 detach() 把旧策略的概率固定下来，不计算它的梯度，因为它只作为参考基准（分母）。
        old_log_probs=torch.log(self.actor(states).gather(1,actions)).detach()
        # 因为有了 clip 机制，PPO 允许我们拿同一批数据反复训练多轮。
        # 这极大地提高了样本利用率，这是 PPO 训练速度快于传统 PG 的主要原因。
        for _ in range(self.epochs):

            log_probs = torch.log(self.actor(states).gather(1, actions))

            # 1. 计算比率 (ratio)
            # 用log计算防止下溢
            # ratio = pi_new(a|s) / pi_old(a|s)
            # exp(log_a - log_b) = a / b
            ratio = torch.exp(log_probs - old_log_probs)

            # 2. 计算两个代理目标 (Surrogate Objectives)
            surr1 = ratio * advantage

            # 3. 截断 (Clipping)
            # 将 ratio 限制在 [1-eps, 1+eps] 范围内，超过会截断
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 4. 计算策略损失 (Actor Loss)
            # 取两者的最小值，这是为了悲观地估计，防止策略更新步子迈得太大
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            # 5. 计算价值损失 (Critic Loss)
            # 也就是 MSE 均方误差，让 Critic 预测的价值越准越好
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 6. 反向传播更新参数
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

def run_PPO():
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.reset(seed=0)

    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()
if __name__ == "__main__":
    run_PPO()