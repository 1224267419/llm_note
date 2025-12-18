![](./note3RLHF.assets/diagram.png)

LLM可以视为上图, 给定一个prompt，大模型会**在 $$t$$时刻生成一个token，然后下一个时刻根据prompt+上一时刻的token再去生成下一个token**，进行自回归的生成，所以可以定义**强化学习**中的各个概念为：

* **动作$$a_t$$：**&#x751F;成的 `token`，动作空间就是整个词表，动作空间大小就是词表大小$$|V|$$

* **策略$$\pi(a_t|s_t)$$：**&#x6839;据当前状态$$s_t$$生成动作`token`$$a_t$$的概率&#x20;

* **状态$$s_t$$：**&#x4E0A;文以及$$t$$时刻前生成的**所有 token concat的 token 序列，**&#x521D;始状态 $$s_0$$就是prompt的token序列

* **状态转移：**&#x8FD9;里的强化学习状态转移是确定性的，定义为**当前状态和动作的concat的token序列为下一个状态**，即$$s_{t+1}=[s_t,a_t]$$

* **奖励$$r_t$$和价值 $$V_t$$：**&#x8FD9;里的定义就是一般强化学习的定义，**即时奖励以及状态价值函数**

## [RLHF](https://arxiv.org/pdf/1706.03741)

### 问题

* 强化学习在许多任务中面临**目标复杂、难以定义奖励函数**的问题，导致**难以将人类实际目标传达给智能体**

* 不正确的、有偏的奖励函数会导致**智能体过分利用(exploit)奖励函数**，产生**reward hacking**问题，即**实际学到的行为与人类期望不符合，甚至有害**

* 奖励函数的设计工程需要**大量的专业人士的精力**

* 现有方法如**逆强化学习和模仿学习在处理复杂行为时存在局限性，直接使用人类反馈作为奖励函数成本过高**

### 目标

用于**解决没有明确定义奖励函数的强化学习**问题，需要满足以下几点：

* 能够解决那些人类**只能识别期望行为**，但**不一定能提供demonstration的任务**

* 允许**非专家用户对智能体进行教导**

* 能够**扩展到大型问题**

* 在**用户反馈方面经济高效

### 方法

将**奖励函数与人类偏好进行拟合**，同时**用RL算法训练一个策略来优化当前预测的奖励函数**。**给人类提供两个智能体行为轨迹的片段(一般来说是视频、动图)，让人给出自己的偏好标签(就是那个片段更好)，而不是提供绝对数值分数**。

![](./note3RLHF.assets/image.png)



* **对比标签：**&#x5BF9;于智能体轨迹片段 $$\sigma^1$$和 $$\sigma^2$$来说，下面的式子表示 $$\sigma^1$$比 $$\sigma^2$$更被人偏好，得到的标签 $$y$$也可以表示如下，0.5代表同等偏好程度。**&#x20;$$s,a$$分别表示智能体的观测/状态和动作**

**&#x20;**$$\sigma^1\succ\sigma^2=
\left(\left(s_{0}^{1}, a_{0}^{1}\right), \ldots,\left(s_{k - 1}^{1}, a_{k - 1}^{1}\right)\right) \succ\left(\left(s_{0}^{2}, a_{0}^{2}\right), \ldots,\left(s_{k - 1}^{2}, a_{k - 1}^{2}\right)\right) 
\\ \\
y = \{0,1,0.5\} \text{  if  } \{\sigma^1\succ\sigma^2, \sigma^2\succ\sigma^1, \sigma^1=\sigma^2\}$$

* **偏好建模：**&#x7531;于RLHF的一个目标是将**奖励函数与人类偏好进行拟合，**&#x5C31;是**利用人类的比较偏好标签来学出一个reward model**，那就涉及到了奖励函数和偏好之间的关联问题，这里给出的方法是，将奖励函数视为解释人类判断的潜在因素，并**假设人类偏好一个片段的概率与潜在奖励在该片段长度上的总和呈指数相关**，基于 **Bradley-Terry 模型**，可以给出**人类偏好片段 $$\sigma^1$$超过 $$\sigma^2$$的概率**：

$$\hat{P}[\sigma^{1} \succ \sigma^{2}] = \frac{\exp \sum \hat{r}(s_{t}^{1}, a_{t}^{1})}{\exp \sum \hat{r}(s_{t}^{1}, a_{t}^{1}) + \exp \sum \hat{r}(s_{t}^{2}, a_{t}^{2})}$$

* **奖励学习：**&#x5F97;到这个偏好建模以及收集到的人类偏好标签之后，就可以简单的使用**二分类的思路来隐式的学习我们的奖励函数**了，损失函数是分类常用的**交叉熵，**&#x7136;后利用这个loss训练优化得到最后的符合人类偏好的奖励函数

 $$ \mathrm{loss}(\hat{r}) = - \mathbb{E}_{(\sigma^{1}, \sigma^{2}, y) \in \mathcal{D}} \left[y(\sigma^1\succ\sigma^2) \log \hat{P}[\sigma^{1} \succ \sigma^{2}] + y(\sigma^2\succ\sigma^1) \log \hat{P}[\sigma^{2} \succ \sigma^{1}]\right]$$

  如果将正样本（被偏好）和负样本（不被偏好）记为 $$\sigma^+,\sigma^-$$，则上述loss可以写成：


 $$ \mathrm{loss}(\hat{r}) = - \mathbb{E}_{(\sigma^{+}, \sigma^{-}, y) \in \mathcal{D}} \left[ \log \hat{P}[\sigma^{+} \succ \sigma^{-}] \right] = \\ - \mathbb{E}_{(\sigma^{+}, \sigma^{-}, y) \in \mathcal{D}} \left[ \log \frac{\exp \sum \hat{r}(s_{t}^{+}, a_{t}^{+})}{\exp \sum \hat{r}(s_{t}^{+}, a_{t}^{+}) + \exp \sum \hat{r}(s_{t}^{-}, a_{t}^{-})}\right]$$

* **策略学习：**&#x5F97;到奖励函数之后就可以应用任何一个强化学习算法去**最大化奖励用来产出相应的策略**了

* **在线学习：**&#x8FD9;篇文章提出的方法是在线RLHF(Online RLHF)，就是**奖励函数和策略学习是交替同时进行的**，伴随着智能体不断的和环境交互产生新的轨迹数据用来给人类打反馈标签

## 研究**总结**

* **算法原理：**算法通过将奖励函数与人类偏好进行拟合，**使智能体的行为朝着符合人类期望的方向发展**。在训练过程中，同时优化策略以最大化预测的奖励，从而在没有明确奖励函数的情况下，让智能体学会做出符合人类偏好的决策

* **反馈方式优势：**

  * **易于提供：**&#x76F8;比提供绝对数值分数，**人类更容易对智能体轨迹片段进行比较**，降低了反馈的难度，使得非专家用户也能更轻松地参与到智能体的训练过程中

  * **信息丰富：**&#x667A;能体轨迹片段包含了一定的行为序列信息，**比单个状态更能反映智能体的行为特点和趋势**，因此在学习人类偏好方面更有帮助，能够**为奖励函数的拟合提供更有价值的信息**

  * **在线反馈的好处：**&#x5728;线收集反馈意味着系统可以**实时获取人类的偏好信息**，并**根据新的反馈及时调整策略和奖励函数**。这样可以**避免系统过度依赖之前学习到的奖励函数**，**防止因奖励函数的不准确性或局限性而导致的不良行为**，从而持续提高系统的性能，使其更好地适应复杂多变的任务环境；缺点就是**实时收集人类偏好标签成本很高**，所以之后在LLM中应用的时候，很多工作都在研究**自动化偏好标签，比如RLAIF，利用大模型代替人类给偏好**



# RLHF+PPO

## RLHF+PPO的简化理解

包括各个模型的loss等

### 参考文献:

https://zhuanlan.zhihu.com/p/677607581 :直觉和实践,这个更好用

https://zhuanlan.zhihu.com/p/7461863937 :原理,结合b站 强化学习数学原理看

https://github.com/wlll123456/study_rlhf 代码

上面两篇文章讲的非常好,建议多看看

![img](note3RLHF.assets/v2-eb250d428d3b9a751d4ba3aeae70e290_1440w-1765942687098.jpg)

LLM可以视为上图, 给定一个prompt，大模型会**在 $$t$$时刻生成一个token，然后下一个时刻根据prompt+上一时刻的token再去生成下一个token**，进行自回归的生成，所以可以定义**强化学习**中的各个概念为：

- **动作$$a_t$$：**&#x751F;成的 `token`，动作空间就是整个词表，动作空间大小就是词表大小$$|V|$$
- **策略$$\pi(a_t|s_t)$$：**&#x6839;据当前状态$$s_t$$生成动作`token`$$a_t$$的概率&#x20;
- **状态$$s_t$$：**&#x4E0A;文以及$$t$$时刻前生成的**所有 token concat的 token 序列，**&#x521D;始状态 $$s_0$$就是prompt的token序列
- **状态转移：**&#x8FD9;里的强化学习状态转移是确定性的，定义为**当前状态和动作的concat的token序列为下一个状态**，即$$s_{t+1}=[s_t,a_t]$$
- **奖励$$r_t$$和价值 $$V_t$$：**&#x8FD9;里的定义就是一般强化学习的定义，**即时奖励以及状态价值函数**

 $A_t$是由我们的语言模型产生的，$R_t$ ，$V_t$ 则分别由另外两个模型来产生 (Actor Model和Critic model)

## 工作流程

![img](note3RLHF.assets/v2-5b0028cc73d9f2aa599b256df24bda83_1440w.jpg)

- 第一步，我们准备一个batch的prompts
- 第二步，我们将这个batch的prompts喂给Actor模型，让它生成对应的responses
- 第三步，我们把prompt+responses喂给我们的Critic/Reward/Reference模型，让它生成用于计算actor/critic loss的数据，按照强化学习的术语，我们称这些数据为经验（experiences）。critic loss我们将在后文做详细讲解，目前我们只把目光聚焦到actor loss上
- 第四步，我们根据这些经验，实际计算出actor/critic loss，然后更新Actor和Critic模型



### 四个角色

![img](note3RLHF.assets/v2-22c2f6fce157dc4385a14f0de50d8136_1440w.jpg)

如上图，**在RLHF-PPO阶段，一共有四个主要模型**，分别是：

- **Actor Model：演员模型**，这就是我们想要训练的目标语言模型 s->a ,强化学习中的策略$\pi$
- **Critic Model：评论家模型**，它的作用是预估总收益 s-> $V_t$
- **Reward Model：奖励模型**，它的作用是计算即时收益 a-> $R_t$
- **Reference Model：参考模型**，它的作用是在RLHF阶段**给语言模型增加一些“约束”**，防止语言模型训歪（朝不受控制的方向更新，效果可能越来越差）

其中:

- **Actor/Critic Model**在RLHF阶段是**需要训练**的（图中给这两个模型加了粗边，就是表示这个含义）；而**Reward/Reference Model**是**参数冻结**的。
- Critic/Reward/Reference Model共同组成了一个“奖励-loss”计算体系（我自己命名的，为了方便理解），我们综合它们的结果计算loss，用于更新Actor和Critic Model

#### Actor Model :  LLM

即我们要训练的LLM,**用SFT阶段产出的SFT模型来对它做初始化**

训练的最终目的是让Actor模型能**产生符合人类喜好的response**。所以我们的策略是，先喂给Actor一条prompt （这里假设batch_size = 1，所以是1条prompt），让它生成对应的response。然后，我们再将“prompt + response"送入我们的“奖励-loss”计算体系中去算得最后的loss，用于更新actor。



#### Reference Model

**我们希望训练出来的Actor模型既能达到符合人类喜好的目的，又尽量让它和SFT模型不要差异太大**。简言之，**我们希望两个模型的输出分布尽量相似**。那什么指标能用来衡量**输出分布的相似度**呢？我们自然而然想到了**KL散度**。

- **对Actor模型**，我们喂给它一个prompt，它正常输出对应的response。那么response中每一个token肯定有它对应的log_prob结果呀，我们把这样的结果记为**log_probs**
- **对Ref模型**，我们把Actor生成的"prompt + response"喂给它，那么它同样能给出每个token的log_prob结果，我们记其为**ref_log_probs** (类比`teacher forcing,用于计算**输出概率的kl散度**)

#### Critic Model

![img](note3RLHF.assets/v2-6d1497cc608b9b5fd059870c7117e381_1440w.jpg)

用于预测期望总收益 $V_{t}$  **，和Actor模型一样，它需要做参数更新**。实践中，Critic Model的设计和初始化方式也有很多种，例如和Actor共享部分参数、从RW阶段的Reward Model初始化而来等等。我们讲解时，和deepspeed-chat的实现保持一致：从RW阶段的Reward Model初始化而来。

**也就是在**  $t$ **时刻，我们给不出客观存在的总收益** $V_t$ **，我们只能训练一个模型去预测它。**
**在RLHF中，我们不仅要训练模型生成符合人类喜好的内容的能力（Actor），也要提升模型对人类喜好量化判断的能力（Critic）**

####  Reward Model（奖励模型）

Reward Model用于计算生成token $A_{t}$ 的即时收益 $R_{t}$ ，它就是RW阶段所训练的奖励模型，在RLHF过程中，**它的参数是冻结的**。


**你可能想问：为什么Critic模型要参与训练，而同样是和收益相关的Reward模型的参数就可以冻结呢？**
这是因为，Reward模型是站在上帝视角的。这个上帝视角有两层含义：

- 第一点，Reward模型是经过和“估算收益”相关的训练的，因此在RLHF阶段它可以直接被当作一个能产生客观值的模型。
- 第二点，Reward模型代表的含义就是“即时收益”，你的token $A_{t}$  已经产生，因此即时收益 $R_{t}$ 自然可以立刻算出。

### Loss设计

#### Actor loss

Actor loss : $actor\_loss = \sum -A_t \log P(A_t | S_t)$

$A_t = R_t + \gamma * V_{t+1} - V_t$

$A_t >0$意味着Critic对Actor当前采取的动作给了正向反馈，因此我们就需要在训练迭代中提高$P(A_t | S_t)$,从而减少loss (上面的$V_t$用动作优势 $A_t$ 替代)
$A_t<0$则相反,不赘述



deepspeed-chat的 $R_t$ 设计：

$R_t =
\begin{cases}
-kl\_ctl \cdot \left( \log \frac{P(A_t|S_t)}{P_{ref}(A_t|S_t)} \right), & t \neq T \\
-kl\_ctl \cdot \left( \log \frac{P(A_t|S_t)}{P_{ref}(A_t|S_t)} \right) + R_t, & t = T
\end{cases}$

$kl\_ctl$用于控制kl散度缩放比例, ( ) 中的内容即两个模型输出的kl散度 
t=T : 在最后一步，模型既要受到“是否偏离原模型”的约束（前半部分），又要接收“**这句话写得好不好”的最终评价**（后半部分）。

为什么只有最后一个时刻的 $R_t$ 被纳入了考量呢？这是因为在Reward模型训练阶段，就是用这个位置的 $R_t$ 来表示对完整的prompt + response的奖励预测（但不妨碍你理解成是执行完 $a_T$  的即时奖励），然后用这个指标来做模型eval的（但是Reward训练阶段算loss时，还是考虑了response部分所有token输出的reward值）。所以到了RLHF的场景下，**其余时刻**的即时奖励，我们就用“Actor**是否遵循了Ref的约束**”来进行评价。

而且, $R_t$的设计并不只有一种,可以尝试把最后一个时刻的 $R_T$ 替换成所有token的即时奖励的平均值。如果站在这个角度理解的话，我们同样也可以尝试在每一个位置的奖励衡量上引入 $R_T$ ,

##### $R_t$计算代码

```python
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        """
        reward_function：计算最终的reward分数
        复习一下几个相关参数的默认值：
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        
        对于batch中的某个prompt来说，它最终的reward分数为：
        (1) 先计算actor和ref_model的logit相似度： -self.kl_ctl * (log_probs - ref_log_probs)
            其实写成self.kl_ctl * (ref_log_probs - log_probs)更好理解些
            这个值越大，说明ref_model对actor生成的结果的认可度越高（即表明rlhf没有训歪），
            没有训歪的情况下我们也应该给模型一些奖励，这个奖励就是self.kl_ctl * (ref_log_probs - log_probs)
            
        （2）由于我们只取最后一个token对应位置的分数作为reward_score，因此我们只需要：
            self.kl_ctl * (ref_log_probs - log_probs)的最后一位 + reward_score
         
         (3) 同时我们对reward_score也做了大小限制，最大不超过self.clip_reward_value（超过统一给成self.clip_reward_value），
             最小不低于-self.clip_reward_value（低于统一给成-self.clip_reward_value）
        
         (4) 最后返回的rewards大小为：（batch_size, 各条数据的长度），对batch中的每条数据来说：
             - response的最后一位：self.kl_ctl * (ref_log_probs - log_probs)的最后一位 + reward_score
             - response的其余位置：self.kl_ctl * (ref_log_probs - log_probs)
        
        """

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        # ---------------------------------------------------------------------------------------------------
        # response开始的位置
        # （因为我们对prompt做过padding处理，因此batch中每个prompt长度一致，也就意味着每个response开始的位置一致）
        # （所以这里start是不加s的，只是一个int）
        start = prompts.shape[1] - 1
        # ---------------------------------------------------------------------------------------------------
        # response结束的位置
        # （因为一个batch中，每个response的长度不一样，所以response的结束位置也不一样）
        # （所以这里end是加s的，ends的尺寸是(batch_size,)
        # ---------------------------------------------------------------------------------------------------
		#从 Response 开始往后有多少个真实的 Token（即 Response 的实际长度）
        ends = start + action_mask[:, start:].sum(1) + 1
        # ---------------------------------------------------------------------------------------------------
        # 对rewards_score 做clip
        # ---------------------------------------------------------------------------------------------------
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        #遍历 Batch 中的每一条数据，只在 Response 的最后一个 Token 上，加上 Reward Model 的打分
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j] # 

        return rewards
```

有了$R_t$,我们可以计算动作优势$A_t$然后引入GAE降低方差(推理具体看上一节RL),有下述代码

##### $A_t$计算代码

```python
 def get_advantages_and_returns(self, values, rewards, start):
        """
        Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        
        没有引入GAE前的t时刻的优势值：
        detal_t = r_t + gamma * V_t+1 - V_t
        其中：
            - r_t表示t时刻的即时收益
            - V_t+1表示未来时刻的预期收益
            - r_t + gamma * V_t+1可理解成t时刻的实际预期收益
            - V_t可理解成t时刻的预估预期收益（是模型，例如critic model自己估算出来的）
        
        引入GAE后的t时刻的优势值：
        A_t = delta_t + gamma * lambda * A_t+1
        粗暴理解为在t时刻时，不仅考虑当下优势，还考虑了未来的优势
        为了知道A_t, 我们得知道A_t+1，所以在本算法中采取了从后往前做动态规划求解的方法，也即：
        假设T是最后一个时刻，则有A_T+1 = 0, 所以有: A_T = delta_T
        知道了A_T, 就可以依次往前倒推，把A_t-1, A_t-2之类都算出来了
        
        引入GAE后t时刻的实际预期收益
        returns_t = A_t + V_t
                  = delta_t + gamma * lambda * A_t+1 + V_t
                  = r_t + gamma * V_t+1 - V_t + gamma * lambda * A_t+1 + V_t
                  = r_t + gamma * (V_t+1 + lambda * A_t+1)
        
        注意，这里不管是advantages还是returns，都只算response的部分
        """
        
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        # 注意这里用了reversed，是采取从后往前倒推计算的方式
        for t in reversed(range(start, length)):
            # 往后挪,用于下一步差分
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            #detal_t= r_t + gamma * V_t+1 - V_t
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            #GAE 优势:A_t= detal_t + gamma * lambda * A_t+1 
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        # 反转列表变回正序
        advantages = torch.stack(advantages_reversed[::-1], dim=1) # 优势
        # Returns = Advantage + Value
        returns = advantages + values[:, start:] # 实际收益
        # values: 预期收益
        return advantages.detach(), returns
```



##### PPO-epoch: 引入新约束

目前的actor_loss

$actor\_loss = -A_t \log P(A_t | S_t)$

其中，
$$
A_t = \left( R_t + \gamma * V_{t+1} - V_t \right) + \gamma * \lambda * A_{t+1}
$$
同时：
- 我们已经对  $R_t$  进行来改造，使其能够衡量Actor模型是否遵从了Ref模型的约束。
- 我们已经对  $A_t$  进行改造，使其不仅考虑了当前时刻的优势，还考虑了未来的优势

1个batch的经验，用于计算ppo-epochs次loss，更新ppo-epochs次Actor和Critic模型 , 所以通过重要性采样 , 即可实现一次经验多次loss , 然后通过clip确保两个模型输出区别不太大,修正后有a_loss

$$actor\_loss = -\min\left( Adv_t * \frac{P(A_t|S_t)}{P_{old}(A_t|S_t)}, \ Adv_t * \text{clip}\left( \frac{P(A_t|S_t)}{P_{old}(A_t|S_t)}, 1-\epsilon, 1+\epsilon \right) \right)$$

综合上面的计算结果,有

```python
    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        """
        logprobs: 实时计算的，response部分的prob（只有这个是随着actor实时更新而改变的）
        old_logprobs：老策略中，response部分的prob （这个是固定的，不随actor实时更新而改变）
        advantages： 老策略中，response部分每个token对应的优势（这个是固定的，不随actor实时更新而改变）
        mask：老策略中，response部分对应的mask情况这个是固定的，不随actor实时更新而改变）
        
        之所以要引入logprobs计算actor_loss，是因为我们不希望策略每次更新的幅度太大，防止模型训歪
        
        self.cliprange: 默认值是0.2
        """
        ## policy gradient loss
        # -------------------------------------------------------------------------------------
        # 计算新旧策略间的KL散度
        # -------------------------------------------------------------------------------------
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        # -------------------------------------------------------------------------------------
        # 计算原始loss和截断loss
        # -------------------------------------------------------------------------------------
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum() # 最后是取每个非mask的response token的平均loss作为最终loss
        return pg_loss
```

#### Critic Loss

Critic Loss应为预测V和实际V的MSE,即 $Critic\_loss = \left( R_t + \gamma * V_{t+1} - V_t \right)^2$ ,那么接下来优化实际收益和预估收益

##### 实际收益(优化后的critic网络输出)优化

$R_t + \gamma * V_{t+1}$ -> $A_t + V_t$

##### 预估收益(优化前critic网络输出)优化



取实际收益和预估收益的MSE做为loss , 加上clip,**防止value剧烈变化**导致loss骤减

```python
def critic_loss_fn(self, values, old_values, returns, mask):
        """
        values: 实时critic跑出来的预估预期收益（是变动的，随着ppo epoch迭代而改变）
        old_values：老critic跑出来的预估预期收益（是固定值）
        returns：实际预期收益
        mask：response部分的mask
        
        self.cliprange_value = 0.2
        """
        ## value loss
        # 用旧的value去约束新的value
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        
        #用fp64计算防止下溢
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        
        # critic模型的loss定义为（预估预期收益-实际预期收益）**2
        #计算两个均方差,并取最大值
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum() # 同样，最后也是把critic loss平均到每个token上
        return vf_loss
```

#### Reward Loss

PPO训练过程中 Reward Model参数被冻结,一般是在SFT model的基础上加上Value Head进行训练

**Reward Model训练的loss如下：**

$$\text{Reward\_loss}=- \mathbb{E}_{(x,y_w,y_l)\sim D} \left[\log\left(\sigma\left(r(x,y_w)-r(x,y_l)\right)\right)\right]$$

其中$$x,y_w,y_l$$分别表示  prompt、 chosen response 和  rejected response


sigmoid函数： $$\sigma(x)=\frac{1}{1+\exp(-x)}$$，所以 $$\sigma(r(x,y_w)-r(x,y_r))=\frac{\exp(r(x,y_w))}{\exp(r(x,y_w))+\exp(r(x,y_l))}$$

最后的reward loss为

$$\text{Reward\_loss}=- \mathbb{E}_{(x,y_w,y_l)\sim D} \left[\log\frac{\exp(r(x,y_w))}{\exp(r(x,y_w))+\exp(r(x,y_l))}\right]$$

```python
class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """
    def forward(self, chosen_reward, reject_reward, margin):
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()
```

和RL中的episode中每个action都求loss想比, LLM中RLHF仅对整个response进行求loss , 
