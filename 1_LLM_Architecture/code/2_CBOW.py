import torch
import torch.nn as nn
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        # 生成嵌入：[num_context_words, embedding_size]
        out = self.linear1(inputs)
        # 对生成的语境embedding做处理
        out=torch.mean(out,dim=0)
        # embedding ->word
        out = self.linear2(out)
        return  out
embedding_dim = 10
cbow_model=CBOW(vocab_size=5,embedding_dim=embedding_dim)
print(cbow_model)