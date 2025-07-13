from turtle import forward
from numpy.core.defchararray import center
import torch
import torch.nn as nn
import torch.nn.functional as F

net  = nn.Sequential(nn.Linear(8,4),nn.ReLU(),
                     nn.Linear(4,2))

x = torch.rand(2,8)
print(net[2].state_dict())
print(net[2].weight.data)
print(net[2].bias.data)
#可以打印参数
print(type(net[2].weight))
print(net[0].weight)
print(net[2].weight.grad == None)  #梯度为None

print(*[(name,param.shape) for name, param in net.named_parameters()])
print(*[(name,param.shape) for name, param in net[2].named_parameters()])#只打印线性层的参数
print([(name,param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['2.bias'])
print(net.state_dict()['2.bias'].data)

def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),
                            nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1())
    return net

rgnet  = nn.Sequential(block2(),nn.Linear(4,1))
print(rgnet)
net = nn.Sequential(nn.Linear(8,4),nn.Linear(4,1))
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean = 0,std = 1)
        nn.init.zeros_(m.bias)

#net.apply(init_normal)
print(net[0].weight.data)
print(net[0].bias.data)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)

#net.apply(init_constant)
print(net[0].weight.data)
print(net[0].bias.data)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)

net[0].apply(xavier)
net[1].apply(init_42)
print(net[0].weight.data)
print(net[0].bias.data)
print(net[1].weight.data)
print(net[1].bias.data)

def my_init(m):
    if type(m) == nn.Linear:
        print("Init",
            *[(name,param.shape) for name,param in m.named_parameters()])
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight.data)
print(net[0].bias.data)


#参数绑定
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
print(net[4])
print(net[2])
print(net[2].weight.data == net[4].weight.data)
net[2].weight.data[0,0] = 114514
print(net[2].weight.data[0] == net[4].weight.data[0])

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x - x.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

net = nn.Sequential(nn.Linear(4,8),CenteredLayer())
y = torch.rand(4,4)
y = net(y)
print(y.mean().item())

class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self,x):
        return F.relu(torch.matmul(x,self.weight.data) + self.bias.data)

dense = MyLinear(5,3)
print(dense.weight)
dense(torch.rand(2,5))


x = torch.arange(4)
y = torch.zeros(4)
torch.save([x,y],'x-File')#存储和读取
mydict = {'x':x, 'y':y}
torch.save(mydict,'mydict')
x2, y2 = torch.load('x-File')
print(x2,y2)
mydict2 = torch.load('mydict')#字典
print(mydict2)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(8,64)
        self.output = nn.Linear(64,10)

    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
x = torch.randn(4,8)
y = net(x)
torch.save(net.state_dict(),'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

y_clone = clone(x)
print(y == y_clone)
