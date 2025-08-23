from mpmath import ker
import torch
from torch import nn
from d2l import torch as d2l

#

def trans_conv(X, k):
    h, w = k.shape
    Y = torch.zeros((X.shape[0]+h-1, X.shape[1]+w-1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i+h, j:j+w] += X[i, j] * k

    return Y

#

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
k = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, k))

#

X, k = X.reshape(1,1,2,2), k.reshape(1,1,2,2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = k
print(tconv(X))

#

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = k
print(tconv(X))

#

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = k
print(tconv(X))

#

X = torch.randn(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
print(tconv(conv(X)).shape == X.shape)

#

X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

Y = d2l.corr2d(X, K)
print(Y)

#

def kernel2matrix(K):
    k, w = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    w[0, :5], w[1, 1:6], w[2, 3:8], w[3, 4:] = k, k, k, k
    return w

w = kernel2matrix(K)
print(w)


#

print(Y == torch.matmul(w, X.reshape(-1)).reshape(2, 2))

#

Z = trans_conv(Y,K)
print(Z.shape)
print(Z)
print(torch.matmul(w.T, Y.reshape(-1)))
print(Z == torch.matmul(w.T, Y.reshape(-1)).reshape(3, 3))

