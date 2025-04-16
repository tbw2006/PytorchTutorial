from numpy import block
import torch
from torch import nn
from d2l import torch as d2l
import train_utils
import matplotlib.pyplot as plt
plt.ion()
#写一个返回dropout之后的x的函数
def dropout_layer(dropout,x):
    assert 0<= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(x)

    mask = torch.randn(x.shape)>dropout
    return x * mask / (1 - dropout)


#写多层感知机
num_inputs = 28*28
num_hiddens_1 = 512
num_hiddens_2 = 256
#num_hiddens_3 = 128
num_outputs = 10
dropout_1 = 0.7
dropout_2 = 0.7
#dropout_3 = 0.7
class net(nn.Module):
    def __init__(self,num_inputs,num_hiddens_1,
    num_hiddens_2,dropout_1,dropout_2):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens_1 = num_hiddens_1
        self.num_hiddens_2 = num_hiddens_2
        self.num_hiddens_3 = num_hiddens_3
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        #self.dropout_3 = dropout_3
        self.lin1 = nn.Linear(num_inputs,num_hiddens_1)
        self.lin2 = nn.Linear(num_hiddens_1,num_hiddens_2)
        self.lin3 = nn.Linear(num_hiddens_2,num_hiddens_3)
        #self.lin4 = nn.Linear(num_hiddens_3,num_outputs)
        self.relu = nn.ReLU()
        self.dropout = dropout_layer


    def forward(self,x):
        is_train = self.training
        x = torch.reshape(x,(-1,num_inputs))
        output = self.relu(self.lin1(x))
        if is_train == True:
            output = self.dropout(self.dropout_1,output)

        output = self.relu(self.lin2(output))
        if is_train == True:
            output = self.dropout(self.dropout_2,output)
        
        ''' output = self.relu(self.lin3(output))
        if is_train == True:
            output = self.dropout(self.dropout_3,output)'''

        output = self.lin3(output)
        return output

batch_size,num_epochs ,lr  = 32,10 ,0.01
train_iter ,test_iter = train_utils.get_data_fashion_mnist(batch_size = batch_size)
net = net(num_inputs,num_hiddens_1,num_hiddens_2,num_outputs,dropout_1,dropout_2)

updater = torch.optim.SGD(net.parameters(),lr)
criterion = nn.CrossEntropyLoss()
train_utils.train(net,train_iter,test_iter,updater,num_epochs,criterion=criterion)
plt.show(block=True)

