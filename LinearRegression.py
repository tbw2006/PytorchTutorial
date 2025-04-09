import matplotlib
import random
from numpy import indices
from sympy import true
import torch
from d2l import torch as d2l
from torch.distributed import batch_isend_irecv

def synthetic_data(w,b,num_samples):
    x = torch.normal(0,1,(num_samples,len(w)))
    y = torch.matmul(x,w)+b
    y += torch.normal(0,0.01,y.shape)#加上一点噪声
    return x,y.reshape(-1,1)        #y要转换成列向量



true_w = torch.tensor([2,-3.4])#这是列向量 2*1
true_b = 4.2
num_samples = 1000
features , labels= synthetic_data(true_w,true_b,num_samples)

print('feature: ',features[0],'\nlabel: ',labels[0])
print(len(features))


#d2l.set_figsize()
#d2l.plt.plot(features[:,1].detach().numpy(),
 #            labels.detach().numpy(),1)
#d2l.plt.show()  # 添加这一行来显示图片


def data_iter(features,labels,batch_size):
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)
    for idx in range(0,num_samples,batch_size):
        batch_indices = torch.tensor(indices[idx:min(idx + batch_size,num_samples)])
        
        yield features[batch_indices],labels[batch_indices]

batch_size = 10
for x,y in data_iter(features,labels,batch_size):
    print(x)
    print(y)
    break


#初始化
w = torch.normal(0,1,(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

def linreg(x,w,b):
    return torch.matmul(x,w) + b


#定义均方损失
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 * 0.5

#定义优化算法
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epochs = 30
net = linreg
criterion = squared_loss

for epoch in range(num_epochs):
    for x, y in data_iter(features,labels,batch_size):
        output = net(x,w,b)
        loss = criterion(output,y)
        loss.sum().backward()
        sgd([w,b],lr,batch_size)        

    #检验
    with torch.no_grad():
        train_loss = criterion(net(features,w,b),labels)

    print(f'epoch: {epoch+1},loss: {float(train_loss.mean()):f}')


print(f'误差： {w.reshape(true_w.shape)- true_w},\n {b-true_b}')
            