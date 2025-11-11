import torch
from torch import nn

class LN(nn.Module):
    def __init__(self, dim, #做LN的维度
                 eps=1e-5,
                 elementwise_affine=True #是否使用可学习缩放
                 ):
        super(LN, self).__init__()
        self.dim= dim
        self.eps = eps
        self.elementwise_affine=elementwise_affine

        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor):  # [b,c,w*h]
        # 需要做LN的维度和输入特征图对应维度的shape相同
        # assert self.dim == x.shape[-len(self.dim):]: 这是一个安全检查。它确保你在 __init__ 中
        # 指定的 dim (例如 (1, 2)) 和输入张量 x 的最后几个维度的大小是匹配的。
        # 例如，如果 x 的形状是 [2, 3, 4] (即 [b=2, c=3, w*h=4])，那么 x.shape[-2:] 就是 (3, 4)。
        # 如果 self.dim 也等于 (3, 4)，则断言成立，程序继续执行。这可以防止因输入维度不匹配而导致的错误。
        assert self.dim == x.shape[-len(self.dim):]  # [-2:]
        # 需要做LN的维度索引
        # 例如，如果 self.dim 的长度是 2，range(len(self.dim)) 会生成 [0, 1]。
        # 循环内，-(i + 1) 会生成 [-1, -2]。
        # 所以 dims = [-1, -2]。这表示我们要在最后两个维度上进行归一化，
        # 这与我们的输入 [b, c, w*h] 和注释相符（在 c 和 w*h 上归一化）
        dims = [-(i + 1) for i in range(len(self.dim))] #「b,c,w*h]维度上取「-1，-2]维度，即「c,w*h]
        # 计算特征图对应维度的均值和方差
        mean = x.mean(dim=dims, keepdims=True)  # [b,1,1]
        mean_x2 = (x ** 2).mean(dim=dims, keepdims=True)  # [b,1,1]
        var = mean_x2 - mean ** 2  # [b,c,1,1]

        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # [b,c,w*h]
        # 线性变换
        if self.elementwise_affine:
            x_norm = self.gamma * x_norm + self.beta  # [b,c,w*h]
        return x_norm

if __name__ == '__main__':
    x = torch.linspace(0, 23, 24, dtype=torch.float32)  # 构造输入层
    x = x.reshape([2, 3, 2 * 2])  # [b,c,w*h]
    # 实例化
    ln = LN(x.shape[1:])
    # 前向传播
    x = ln(x)
    print(x.shape)
