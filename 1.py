from datasets.formatting.torch_formatter import torch
import torch
a=torch.tensor([1.,2.,3.,4.,5.])
def online_softmax(x):
    
    m=max(x)
    d=torch.sum((x-m).exp())

    x=(x-m).exp()/d

    return x

print(torch.softmax(a,dim=-1))
print(online_softmax(a))