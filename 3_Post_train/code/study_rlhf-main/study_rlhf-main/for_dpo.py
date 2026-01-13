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
# # 手写DPO
# 公式推导https://mp.weixin.qq.com/s/S72LO26IsZ8AED8sQKIWnQ
#
# dpo主要复用ppo里的函数，
#
# dpo的数据格式可以参考llama-factoryhttps://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html#id10
#
# 有chosen以及reject，分别对应正样本和负样本
#
# 简要总结：
#
# 有chosen数据以及reject数据，分别对应正样本和负样本
#
# 有ref模型以及actor模型，
#
# ref模型以及actor模型都使用相同的结构，但是参数不同
#
# x1 = actor模型对chosen数据得到log（softmax（logits（chosen数据））） 减去 ref模型对chosen数据得到log（softmax（logits（chosen数据）））
#
# x2 = actor模型对reject数据得到log（softmax（logits（reject数据））） 减去 ref模型对reject数据得到log（softmax（logits（reject数据）））
#
# loss = -F.logsigmoid(beta * （x1 - x2） ) * label
#
# beta是超参数
#
# label代表我们要关注response部分的loss

# %% [markdown]
# 现在初始ref模型以及actor模型

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
    vocab_size=50257,
    n_embd=hidden_size,
    n_inner=intermediate_size,
    n_layer=num_hidden_layers,
    n_head=num_attention_heads
)

# 初始化 GPT - 2 模型
model = GPT2LMHeadModel(config)
ref_model = GPT2LMHeadModel(config)
ref_model.eval()

# %% [markdown]
# ## 提取logprobs

# %%
import torch.nn.functional as F
import torch

def get_logits(model, inputs):
    # 得到logits
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs.logits
    return logits


def get_logprobs(model, response_inputs, index_label):
    # 得到logprobs
    logits = get_logits(model, response_inputs)
    # F.log_softmax() 是先进行softmax运算然后再取对数（log）
    all_token_logprobs = F.log_softmax(logits, dim=-1)
    # 使用torch.gather() 从logprobs中收集response的值
    gathered = torch.gather(all_token_logprobs, 2, index_label.unsqueeze(2))
    # 去掉最后一个维度
    response_logprobs = gathered.squeeze(-1)
    return response_logprobs


# %% [markdown]
# 初始化chosen以及reject

# %%
batch_size = 2
length_x = 5
max_new_tokens = 5
prompt = torch.randint(0, vocab_size, (batch_size, length_x))
chosen_response = torch.randint(0, vocab_size, (batch_size, length_x + max_new_tokens))
reject_response = torch.randint(0, vocab_size, (batch_size, length_x + max_new_tokens))
attention_mask = torch.ones(batch_size, length_x+max_new_tokens)
label =  torch.tensor([[0, 0, 0, 0,  0, 1,   1,  1,  1,  1]], dtype=torch.bool)

# %%
prompt

# %%
chosen_response

# %%
reject_response

# %%
attention_mask

# %%
label

# %%
x_chosen = {'input_ids':chosen_response, 'attention_mask':attention_mask}
x_rejected = {'input_ids':reject_response, 'attention_mask':attention_mask}

# %% [markdown]
# # 计算ref模型和actor模型的logprobs
#
# chosen_response
# reject_response

# %%
probs_chosen_ref = get_logprobs(ref_model, x_chosen, chosen_response)
probs_chosen = get_logprobs(model, x_chosen, chosen_response)
probs_rejected_ref = get_logprobs(ref_model, x_rejected, reject_response)
probs_rejected = get_logprobs(model, x_rejected, reject_response)

# %%
probs_chosen

# %%
probs_chosen_ref

# %% [markdown]
# ## 计算loss

# %% [markdown]
# ![for_output/image/Snipaste_2025-03-31_20-41-35.png](attachment:image.png)

# %%
import torch.nn.functional as F

beta = 0.01

x = (probs_chosen - probs_rejected) - (probs_chosen_ref - probs_rejected_ref)
loss = -F.logsigmoid(beta * x ) * label
loss = loss.sum(-1)/attention_mask.sum()
print(loss)

# %%
