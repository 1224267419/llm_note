import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, K):
        self.K = K
        # 每根拉杆的获奖概率
        self.p = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.p)
        self.best_prob = self.p[self.best_idx]

    def step(self, a):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.p[a]:
            return 1
        return 0
np.random.seed(1)
K=10
bb = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
      (bb.best_idx, bb.best_prob))

class Solver:
    # 实现上述的多臂老虎机的求解方案
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) #每根拉杆尝试次数
        self.regret = 0 #当前步的积累懊悔
        self.actions = [] #每步的拉杆选择
        self.regrets = []#每步的累计懊悔
    def update_regret(self, k):
        # 更新当前步的累计懊悔,k为本次选择的拉杆编号
        self.regret+=self.bandit.best_prob-self.bandit.p[k]
        self.regrets.append(self.regret)
    def run_one_step(self):
        # 选择了哪一根拉杆,由决策策略决定
        raise NotImplementedError
    def run(self,num_steps):
        # 运行num_steps步
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k]+=1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    # 贪心算法
    def __init__(self, bandit, epsilon=0.01,init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob]*self.bandit.K)
    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            # 随机选择一根拉杆
            k = np.random.randint(self.bandit.K)
        else:
            # 选择当前奖励估值最大的拉杆
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)  # 得到本次动作的奖励
        # 修正当前期望估计值
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


# np.random.seed(1)
# epsilon_greedy_solver = EpsilonGreedy(bb, epsilon=0.01)
# epsilon_greedy_solver.run(5000)
# print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
# plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
#
# # 更换不同的eps,得到的斜率不同,但均为线性增长
# np.random.seed(0)
# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
# epsilon_greedy_solver_list = [
#     EpsilonGreedy(bb, epsilon=e) for e in epsilons
# ]
# epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
# for solver in epsilon_greedy_solver_list:
#     solver.run(5000)
#
# plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

class EpsilonGreedyWithTime(Solver):
    # 随时间改变e的贪心算法,e=1/t
    def __init__(self, bandit, epsilon=0.01,init_prob=1.0):
        super(EpsilonGreedyWithTime, self).__init__(bandit)
        self.count=1
        # 初始化所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob]*self.bandit.K)
    def run_one_step(self):
        if np.random.rand() < (1/self.count):
            # 随机选择一根拉杆
            k = np.random.randint(self.bandit.K)
        else:
            # 选择当前奖励估值最大的拉杆
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)  # 得到本次动作的奖励
        # 修正当前期望估计值
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        self.count+=1
        return k
# regret接近O(logt)
np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedyWithTime(bb, epsilon=0.01)
epsilon_greedy_solver.run(500)
print('epsilon衰减-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)
coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])
