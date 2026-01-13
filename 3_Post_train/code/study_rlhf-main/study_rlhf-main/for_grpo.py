# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 手撕grpo
#
# 截图素材：https://zhuanlan.zhihu.com/p/24816372882 https://zhuanlan.zhihu.com/p/657693775
#
# 这个notebook的reward由规则给出，比如format，以及accury
#
# 只有actor模型，ref模型，没有critic模型
#
# 同时reward是基于规则给出的，所以没有rēward模型

# %% [markdown]
# 定义参数配置

# %%
vocab_size = 50257
hidden_size = 128
intermediate_size = 256
num_hidden_layers = 2
num_attention_heads = 4
batch_size = 2
length_x = 5
max_new_tokens = 5
grpo_samples_nums = 2 # GRPO 采样数量，有的框架会把一张卡分给vllm，来让vllm加速生成

# %% [markdown]
# 初始化ref 和actor模型

# %%
import torch
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(1)

# 定义参数
vocab_size = 10
hidden_size = 128
intermediate_size = 256
num_hidden_layers = 2
num_attention_heads = 4

# 加载模型配置
config = GPT2Config(
    vocab_size=vocab_size,
    n_embd=hidden_size,
    n_inner=intermediate_size,
    n_layer=num_hidden_layers,
    n_head=num_attention_heads
)

# 初始化 GPT - 2 模型
model = GPT2LMHeadModel(config)
model_ref = GPT2LMHeadModel(config)
model.config.pad_token_id = model.config.eos_token_id
model_ref.config.pad_token_id = model_ref.config.eos_token_id


# %% [markdown]
# 定义奖励函数
#
# 参考https://huggingface.co/docs/trl/main/en/grpo_trainer#looking-deeper-into-the-grpo-method

# %%
def reward_func_len(completions, **kwargs):
    """Reward function that gives higher scores to longer completions."""
    return [float(len(completion)) for completion in completions]


# %%
prompts = ["The sky is", "The sun is"]
completions = [" blue.", " in the sky."]
print(reward_func_len(prompts=prompts, completions=completions))

# %%
import re

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

prompts = [
    [{"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
    [{"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
]
completions = [
    [{"role": "assistant", "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>"}],
    [{"role": "assistant", "content": "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8."}],
]
format_reward_func(prompts=prompts, completions=completions)

# %%
import re

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

prompts = ["Problem: Solve the equation $2x + 3 = 7$. Solution:", "Problem: Solve the equation $3x - 5 = 10$."]
completions = [r" The solution is \boxed{2}.", r" The solution is \boxed{6}."]
ground_truth = ["2", "5"]
reward_func(prompts=prompts, completions=completions, ground_truth=ground_truth)

# %% [markdown]
# 以下是使用 GRPOTrainer 中多个奖励函数的示例。在这个例子中，我们定义了两个特定任务的奖励函数： math_reward_func 和 coding_reward_func 。 math_reward_func 奖励基于正确性的数学问题，而 coding_reward_func 奖励基于解决方案是否工作的编程问题。
#
# 不过check_math_solution 以及 test_code_solution要自己写

# %%
from datasets import Dataset
from trl import GRPOTrainer

# Define a dataset that contains both math and coding problems
dataset = Dataset.from_list(
    [
        {"prompt": "What is 2+2?", "task": "math"},
        {"prompt": "Write a function that returns the sum of two numbers.", "task": "code"},
        {"prompt": "What is 3*4?", "task": "math"},
        {"prompt": "Write a function that returns the product of two numbers.", "task": "code"},
    ]
)

# Math-specific reward function
def math_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "math":
            # Calculate math-specific reward
            correct = check_math_solution(prompt, completion)
            reward = 1.0 if correct else -1.0
            rewards.append(reward)
        else:
            # Return None for non-math tasks
            rewards.append(None)
    return rewards

# Coding-specific reward function
def coding_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "coding":
            # Calculate coding-specific reward
            works = test_code_solution(prompt, completion)
            reward = 1.0 if works else -1.0
            rewards.append(reward)
        else:
            # Return None for non-coding tasks
            rewards.append(None)
    return rewards

# # Use both task-specific reward functions
# trainer = GRPOTrainer(
#     model="Qwen/Qwen2-0.5B-Instruct",
#     reward_funcs=[math_reward_func, coding_reward_func],
#     train_dataset=dataset,
# )

# trainer.train()


# %% [markdown]
# ## grpo把critic model使用基于规定的优势函数计算来替换掉了
#
# 可以看到我们需要先准备一批oi才能计算出优势函数

# %% [markdown]
# ![GRPO.png](attachment:GRPO.png)

# %% [markdown]
# 一般情况下，比如trl是可以通过vllm来生成一批数据，我们复制一波输入然后model generate一批就行，在此刻就随便搞一波就行，

# %%
def get_response(model, prompt, max_new_tokens):
    inputs = {'input_ids': prompt}  # ignore mask，好像不需要mask
    y = model.generate(**inputs,
                       max_new_tokens=max_new_tokens,
                       # forced_eos_token_id=True
                       )
    return y


# %% [markdown]
# 这个notebook真实数据不好写（带think以及answer标签），就写个大概

# %%
input_x = [{"prompt": "What is 2+2?", "task": "math"},
        {"prompt": "Write a function that returns the sum of two numbers.", "task": "code"},
        {"prompt": "What is 3*4?", "task": "math"},
        {"prompt": "Write a function that returns the product of two numbers.", "task": "code"}]

# %% [markdown]
# 调用一个get_response，生成一批output_x

# %%
# 假设模型生成的response
output_x = ["<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>",
            "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8.",
            "<think>The product of 3 and 4 is 12.</think><answer>3 * 4 = 12</answer>",
            "The product of 3 and 4 is 12. So 3 * 4 = 12."]

# %% [markdown]
# 我们现在有一组数据后就要得到reward

# %%
len_rewards = reward_func_len(completions=output_x)


# %%
def reward_func_format(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
format_rewards = reward_func_format(completions=output_x)


# %% [markdown]
# 对每组奖励进行计算优势函数
#

# %% [markdown]
# $$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$ 

# %%
def grpo_advantage(rewards):
    epsilon = 0.01
    rewards = torch.tensor(rewards) 
    A = (rewards - rewards.mean()) / (rewards.std() + epsilon)
    return A

advantage_len = grpo_advantage(len_rewards)
print(advantage_len)
advantage_format = grpo_advantage(format_rewards)
print(advantage_format)

# %% [markdown]
# 剩下的流程其实与ppo相同，不太想写了

# %% [markdown]
# $$\mathbb{D}_{\text{KL}}\left[\pi_\theta \|\pi_{\text{ref}}\right] = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1,
# $$

# %%
prompt = torch.randint(0, vocab_size, (batch_size, length_x))
response = torch.randint(0, vocab_size, (batch_size, length_x + max_new_tokens))

# %%
attention_mask = torch.ones(batch_size, length_x+max_new_tokens)
attention_mask[:, :length_x] = 0
print(attention_mask)

# %%
import torch.nn.functional as F

def get_logits(model, input_ids):
    # 得到logits
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    return logits

def get_logprobs(model, response, attention_mask):
    # 得到logprobs
    logits = get_logits(model, response)
    # F.log_softmax() 是先进行softmax运算然后再取对数（log）
    all_token_logprobs = F.log_softmax(logits, dim=-1)
    # 使用torch.gather() 从logprobs中收集response的值
    gathered = torch.gather(all_token_logprobs, 2, response.unsqueeze(2))
    # 去掉最后一个维度
    response_logprobs = gathered.squeeze(-1)
    return response_logprobs


# %%
get_logprobs(model, response, attention_mask)


# %%
def grpo_kl(pi, pi_ref):
    x1 = pi_ref.exp() / pi.exp()
    x2 = pi_ref - pi
    return x1 - x2 - 1

pi = get_logprobs(model, response, attention_mask)
pi_ref = get_logprobs(model_ref, response, attention_mask)
grpo_kl(pi, pi_ref)

# %% [markdown]
# loss
#
# $$
# \mathcal{L}_{\text{GRPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min \left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})} \hat{A}_{i,t}, \, \text{clip}\left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right],
# $$

# %% [markdown]
# ![GRPO-loss.png](attachment:GRPO-loss.png)

# %% [markdown]
# 超参数
#
# epslion = 0.01
#
# beta = 0.1
#
#

# %%
group_num, len_oi = pi.shape

# %%
len_oi = len_oi - length_x

# %%
len_oi

# %%
len_oi = torch.tensor([len_oi] * group_num, dtype = torch.long)
len_oi



# %%
attention_mask

# %%
epslion = 0.01
beta = 0.01

# %%
test_rewards = [100,66]
test_rewards = torch.tensor(len_rewards,dtype = torch.float)
advantage_test = grpo_advantage(test_rewards)

# %%
advantage_test = advantage_test.unsqueeze(dim = 1) 

ratio = pi.exp() / pi_ref.exp()
ratio_clip = torch.clamp(ratio, 1 - epslion, 1 + epslion)

policy_gradient = torch.minimum(ratio * advantage_test , ratio_clip * advantage_test)
kl = grpo_kl(pi, pi_ref)

# %%
ratio_clip

# %%
policy_gradient

# %%
kl

# %%
loss = (policy_gradient -  beta * kl) * attention_mask
loss

# %%
len_oi.unsqueeze(dim = 1)


# %%
def grpo_loss(pi, pi_old, pi_ref, advantage, length_x, mask):
    epslion = 0.01
    beta = 0.01

    advantage = advantage.unsqueeze(dim = 1) 

    ratio = pi.exp() / pi_old.exp()
    ratio_clip = torch.clamp(ratio, 1 - epslion, 1 + epslion)

    policy_gradient = torch.minimum(ratio * advantage , ratio_clip * advantage)
    kl = grpo_kl(pi, pi_ref)

    group_num, len_oi = pi.shape  
    
    len_oi = len_oi - length_x
    len_oi = torch.tensor([len_oi] * group_num, dtype = torch.long)

    loss = (policy_gradient -  beta * kl) * mask
    loss = (- 1 / group_num ) * loss / len_oi.unsqueeze(dim = 1)
    loss = loss.sum()

    return loss



# %%
pi_logprob.shape

# %% [markdown]
# 因为response是自己定义的
#
# 但之前的rewards是基于output_x的，所以需要转换
#
# 随便设置一个rewards

# %%
response

# %%
len_rewards

# %%
len_rewards = [100,66]
len_rewards = torch.tensor(len_rewards,dtype = torch.float)

# %%
pi_logprob = get_logprobs(model, response, attention_mask)
pi_old_logprob = get_logprobs(model, response, attention_mask)
print(pi_logprob == pi_old_logprob)
pi_ref_logprob = get_logprobs(model_ref, response, attention_mask)

len_advantage = grpo_advantage(len_rewards)
loss = grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, len_advantage, length_x, attention_mask)     # 应该是 [B]（然后才可以 unsqueeze 到 [B, 1]）

# %%
loss

# %% [markdown]
# ## grpo采样

# %%
