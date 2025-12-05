import copy
import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""

    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    # P[s][a]=(p, next_state, reward, done)
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


class PolicyIteration:
    """ 策略迭代算法 """

    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta  # 收敛阈值，用来判断价值函数是否还在剧烈变化
        self.gamma = gamma  # 折扣因子，决定了我们看重未来奖励的程度
        # 初始value=0
        self.v = [0] * self.env.ncol * self.env.nrow
        # 初始策略为随机
        self.pi = [[0.25, 0.25, 0.25, 0.25]] * self.env.ncol * self.env.nrow

    def policy_evaluation(self):
        cnt = 1  # 计数器
        while True:
            max_diff = 0 # 用来记录本轮迭代中，V值变动的最大幅度
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 存放本轮计算出的 V 值
                # for a in range(self.env.n_action):
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:  # 状态s下，执行动作a，得到结果res
                        # (1-done) 的作用是：如果 next_state 是终点/悬崖(done=True)，则没有未来价值，后面一项归零。
                        prob, next_state, reward, done = res
                        # Q(s,a) = 概率 * (即时奖励 + 折扣因子 * 下一个状态的价值)
                        qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                        # 本章环境比较特殊,奖励和下一个状态有关,所以需要和状态转移概率相乘
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self):  # 策略提升
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    prob, next_state, reward, done = res
                    qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            #每个state取最大的Q值(greedy
            maxq = max(qsa_list)

            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了相同的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        while 1:
            self.policy_evaluation()
            # 深拷贝,用于比较
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break

class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = [0] * self.env.ncol * self.env.nrow
        # 不必显式计算pi,因此可以最后再给出
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]
    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        prob, next_state, reward, done = res
                        qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)
                # === 核心区别 ===
                # 在策略迭代中，这里是 sum(pi * qsa)（加权平均）
                # 在价值迭代中，这里是 max(qsa_list)（直接取最大值）
                # 这意味着：我们强制认为智能体之后会选最好的那条路走。
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("价值迭代进行%d轮后完成" % cnt)
        self.get_policy()
    def get_policy(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    prob, next_state, reward, done = res
                    qsa += prob * (reward + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq=max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]

# 对于打印出来的动作，我们用^o<o表示等概率采取向左和向上两种动作，
# ooo>表示在当前状态只采取向右动作
def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

def print_policy():
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])

def print_value():
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])
def ice_hold():
    import gym
    env = gym.make("FrozenLake-v1")  # 创建环境
    env = env.unwrapped  # 解封装才能访问状态转移矩阵P
    env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境

    holes = set()
    ends = set()
    for s in env.P:
        for a in env.P[s]:
            for s_ in env.P[s][a]:
                if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                    ends.add(s_[1])
                if s_[3] == True:
                    holes.add(s_[1])
    holes = holes - ends
    print("冰洞的索引:", holes)
    print("目标的索引:", ends)

    for a in env.P[14]:  # 查看目标左边一格的状态转移信息
        print(env.P[14][a])

    print('-'*30)
    action_meaning = ['<', 'v', '>', '^']
    theta = 1e-5
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

if __name__ == '__main__':
    # print_policy()
    # print_value()
    ice_hold()