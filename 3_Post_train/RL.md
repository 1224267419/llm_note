# [强化学习的数学原理](https://www.bilibili.com/video/BV1sd4y167NS/?share_source=copy_web&vd_source=9e952e3695aa7bfc9ff110afee9f3d34)

视频20h,总计约40h

## 总览

![image-20251125210326171](./RL.assets/image-20251125210326171.png)

强化学习目标:求解最优策略,即贝尔曼最优公式

## 1.concepts概念

 State:the state of the agent with respect to the environment
State space<img src="./RL.assets/image-20251125213632295.png" alt="image-20251125213632295" style="zoom: 50%;" />状态的集合
Action:For each state,possible action
Action space of state:<img src="./RL.assets/image-20251125213802105.png" alt="image-20251125213802105" style="zoom:50%;" />对于当前state的Action Set
State transition:<img src="./RL.assets/image-20251125214020619.png" alt="image-20251125214020619" style="zoom:50%;" />like this
tabular representation:like this![image-20251125214322006](./RL.assets/image-20251125214322006.png)
state transition prosibility:$\begin{aligned}&p(s_2|s_1,a_2)=1\\&p(s_i|s_1,a_2)=0\quad\forall i\neq2\end{aligned}$ :at  s1,use a2, arrive s2 posilibity  given is a determined example,but it could be stochastic(随机的)
**Policy**:tell agent to what action to take at a state![image-20251125215420672](./RL.assets/image-20251125215420672.png)
the green one is policy
**mathematical representation数学表示**:更normal,其中pi是条件概率,如图所示,$\pi(a_i|s_j)$即在任意状态下,任意action的概率,同一个$s_j$下的条件概率之和应为1 
![image-20251125215729626](./RL.assets/image-20251125215729626.png)

**Reward**: a real number we get after taking an action. human-machine interface 

- A **positive** reward represents **encouragement** to take such actions.
- A **negative** reward represents **punishment** to take such actions.
- reward could be stochastic随机

​	Questions:

- What about a zero reward? No punishment.
- Can positive mean punishment? Yes.

trajectory and return  :智能体与环境在任意时间段内的交互序列
	 A trajectory is a state-action-reward chain:
	 $$s_1 \xrightarrow[a_2]{r=0} s_2 \xrightarrow[a_3]{r=0} s_5 \xrightarrow[a_3]{r=0} s_8 \xrightarrow[a_2]{r=1} s_9$$  
The return of this trajectory is the sum of all the rewards collected along the trajectory:
	 $$\text{return} = 0 + 0 + 0 + 1 = 1$$
	**better policy has a greater return**
**discounted return** :return with discount rate
 	$$ \begin{align*} \text{discounted return} &= 0 + \gamma 0 + \gamma^2 0 + \gamma^3 1 + \gamma^4 1 + \gamma^5 1 + \dots \\ &= \gamma^3(1 + \gamma + \gamma^2 + \dots) = \gamma^3 \frac{1}{1-\gamma}. \end{align*} $$
	对于这个discounted return,他的值从之前的发散变成了收敛(有上界),这显然更有助于我们研究

episode**回合**:从任务开始到结束的**完整(different to trajectory)交互过程**

###### MDP

对于MDP,其包含的要素有

1. Sets:
   - State: S
   - Action: A(s) s∈S
   - Reward : R(s,a)
2. Probability distribution
   - State transition probability
   - reward probability

### Key elements of MDP:

- **Sets:**  - **State:** the set of states $S$  
-  **Action:** the set of actions  $\mathcal{A}(s)$  is associated for state  $s \in S$.  
- **Reward:** the set of rewards$ \mathcal{R}(s,a)$ . 
-  **Probability distribution:**  
- **State transition probability:** at state  $s$ , taking action  $a$ , the probability to transit to state  $s'$  is  $p(s'|s,a) $  
- **Reward probability:** at state  $s $, taking action  $a$ , the probability to get reward  $r$  is  $p(r|s,a)$  
- **Policy:** at state  s , the probability to choose action  a  is  $\pi(a|s)$  
- **Markov property:** memoryless property  $$  \begin{align*}  p(s_{t+1}|a_{t+1}, s_t, \dots, a_1, s_0) &= p(s_{t+1}|a_{t+1}, s_t), \\  p(r_{t+1}|a_{t+1}, s_t, \dots, a_1, s_0) &= p(r_{t+1}|a_{t+1}, s_t).  \end{align*}  $$  All the concepts introduced in this lecture can be put in the framework in MDP.

## 2.Bellman equation贝尔曼公式



### 2.1 Motivating examples动机

### 2.2 State value

### 2.3 Bellman equation : Derivation

### 2.4 Bellman equation : Matrix-vector form

### 2.5 Bellman equation : Solve the state values

### 2.6 Action value

### 2.7 Summary

第一部分我们推导贝尔曼

[TODO](https://www.bilibili.com/video/BV1sd4y167NS?spm_id_from=333.788.player.switch&vd_source=82d188e70a66018d5a366d01b4858dc1&p=4)6:32





# CS285
