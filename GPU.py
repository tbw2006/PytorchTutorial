import torch
from torch import nn

print(torch.device('cpu')  , torch.cuda.device('cuda')   , torch.cuda.device('cuda:0'))

print(torch.cuda.device_count())

def try_gpu(i = 0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    if torch.cuda.device_count() >= 1:
        device = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        return device
    return torch.device('cpu')


x = torch.tensor([1.0,2,3],device=try_gpu())
print(x.device)
y  = torch.rand(1,2,device=try_gpu())
print(y.device)
z = x.cuda()

net = nn.Sequential(nn.Linear(3,1))
net = net.to(try_gpu())
print(net(x))
print(net[0].weight.data.device)