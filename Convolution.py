from turtle import forward
import torch
from torch import nn as nn
from d2l import torch as d2l
from torch.nn.modules import conv

def corr2d(x,k):
    h,w =  k.shape
    y = torch.zeros(x.shape[0] - h + 1, x.shape[1] - w + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] = (k * x[i:i+h,j:j+w]).sum()

    return y

# 实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return corr2d(x,self.weight) + self.bias

X = torch.ones(6,8)
X[:,2:6] = 0
print(X)

K = torch.tensor([[1.0,-1.0]])
Y = corr2d(X,K)
print(Y)
print(corr2d(X.t(),K))

conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias  = False)

X = X.reshape((1,1,6,8))
Y = Y.reshape(1,1,6,7)  # also okay

for i in range(50):
    Y_hat = conv2d(X)
    l = (Y_hat - Y)**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data -= 3e-2 * conv2d.weight.grad
    if (i+1) % 2 == 0:
        print(f'batch {i+1}: loss {l.sum():.3f}')

print(conv2d.weight.data)

