import torch
import torch.nn as nn
from jieba.lac_small.predict import batch_size


class MLA(nn.modules):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MLA, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads=num_heads
        self.head_dim = hidden_size // num_heads

        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, input_size)

    def forward(self, x, attention_mask=None):
        batch_size = x.shape[0]
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        query=self.split_heads(query)
        key =self.split_heads(key)
        value=self.split_heads(value)

        # q@k /sqrt(d_k)
        attention_score = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(self.head_dim))
        if attention_mask != None:
            # 屏蔽右上三角的点
            attention_score = attention_score.masked_fill(attention_mask == 0, float('-inf'))
            # 对注意力分数进行归一化
        attention_score = torch.softmax(attention_score, dim=-1)
        output = torch.matmul(attention_score, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        output = self.W_o(output)
        return output

    def split_heads(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


import torch
import torch.nn as nn


class MQA(nn.modules):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MQA, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_liner = nn.Linear(input_size, hidden_size)
        self.v_liner = nn.Linear(input_size, hidden_size)
        self.k_liner = nn.Linear(input_size, hidden_size)
        self.out_liner = nn.Linear(hidden_size, input_size)

    def split_heads(self, x,head_num=0):
        batch_size = x.shape[0]
        # 调用多头
        if head_num==0:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:#不使用多头
            return x.view(batch_size, -1, head_num,       self.head_dim).transpose(1, 2)

    def forward(self, x, attention_mask=None):
        batch_size = x.shape[0]
        query = self.q_liner(x)
        key = self.k_liner(x)
        value = self.v_liner(x)

        query = self.split_heads(query)
        key = self.split_heads(key,1)
        value=self.split_heads(value,1)


        key=key.expand(-1,self.num_heads,-1,-1)
        value=value.expand(-1,self.num_heads,-1,-1)