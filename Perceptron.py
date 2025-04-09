import torch
from torch import nn
from d2l import torch as d2l

from SoftmaxRegression import cross_entropy, get_data_fashion_mnist,sgd, train

batch_size = 256
train_iter , test_iter = get_data_fashion_mnist(batch_size=batch_size)

num_inupts = 28*28
num_hiddens = 256
num_outputs = 10

w1 = torch.randn(num_inupts,num_hiddens,requires_grad=True)#换成0矩阵会怎么样
b1 = torch.randn(num_hiddens,requires_grad=True)
w2 = torch.randn(num_hiddens,num_outputs,requires_grad=True)
b2 = torch.randn(num_outputs,requires_grad=True)
params = [w1,b1,w2,b2]

num_epochs = 10
lr = 0.1

def ReLU(x):
    a = torch.zeros_like(x)
    return max(a,x)

criterion = cross_entropy
updater = sgd(params = params,lr=lr)

def net(x,w1,b1,w2,b2,activation=ReLU):
    outputs = torch.matmul(w1,x)+b1
    outputs = activation(outputs)
    outputs = torch.matmul(w2,outputs)+b2
    return outputs

train(net = net,updater=updater,train_iter=train_iter,
test_iter = test_iter,num_epochs=num_epochs)

