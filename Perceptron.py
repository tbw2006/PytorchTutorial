import torch
import math
from torch import nn
from d2l import torch as d2l

from train_utils import train,sgd,get_data_fashion_mnist,cross_entropy
import importlib
import train_utils
importlib.reload(train_utils)
batch_size = 256
train_iter , test_iter = get_data_fashion_mnist(batch_size=batch_size)

num_inupts = 28*28
num_hiddens = 256
num_outputs = 10

'''w1 = torch.randn(num_inupts,num_hiddens,requires_grad=True)#换成0矩阵会怎么样
b1 = torch.zeros(num_hiddens,requires_grad=True)
w2 = torch.randn(num_hiddens,num_outputs,requires_grad=True)
b2 = torch.zeros(num_outputs,requires_grad=True)'''
b1 = torch.zeros(num_hiddens,requires_grad=True)
b2 = torch.zeros(num_outputs,requires_grad=True)
w1 = torch.randn(num_inupts, num_hiddens, requires_grad=True) 
w1.data *=  math.sqrt(2.0 / num_inupts)
w2 = torch.randn(num_hiddens, num_outputs, requires_grad=True) 
w2.data *=  math.sqrt(2.0 / num_hiddens)
params = [w1,b1,w2,b2]

num_epochs = 10
lr = 0.1

def ReLU(x):
    a = torch.zeros_like(x)
    return torch.max(a,x)

#criterion = cross_entropy
#updater = sgd(params = params,lr=lr)#换成官方的试试
updater = torch.optim.SGD(params=params,lr=lr)
criterion = nn.CrossEntropyLoss()
class MLP:
    def __init__(self,w1,b1,w2,b2,activation):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.activation = activation

    def __call__(self, x):
        #print("call")
        x = x.reshape(-1,num_inupts)
        h = torch.matmul(x,self.w1) +self.b1
        h = self.activation(h)
        return torch.matmul(h,self.w2)+self.b2

net = MLP(w1,b1,w2,b2,ReLU)
train(net = net,train_iter=train_iter,
test_iter = test_iter,updater=updater,
num_epochs=num_epochs,params=params,criterion=criterion)




