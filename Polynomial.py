from cProfile import label
import math
import numpy as np
from sympy import false, true
from sympy.solvers.diophantine.diophantine import length
import torch
from torch import nn
from d2l import torch as d2l

import test
import train_utils
import matplotlib.pyplot as plt
plt.ion()  # 打开交互模式
#来使用多项式来看过拟合，欠拟合

max_degree = 20 #最大的多项式次数
num_train ,num_test = 100,100
true_w = np.zeros(max_degree)
true_w[0:4] = [5,1.2,-3.4,5.6]
features = np.random.normal(size = num_train+num_test)
np.random.shuffle(features)
poly_features = np.zeros(shape= (num_train+num_test,max_degree))#num_train * max_degree,(i,j) = 第i+1个样本，x的j次方
for i in range(num_train+num_test):
    poly_features[i,:] = np.power(features[i],range(max_degree))
for i in range(max_degree):
    poly_features[:,i] /= math.factorial(i) 
labels = np.dot(poly_features,true_w)
labels += np.random.normal(scale = 0.1,size = labels.shape)

true_w,features,poly_features,labels = [torch.tensor(x,dtype = torch.float32)
for x in [true_w,features,poly_features,labels]]
#看看数据
'''
print(true_w)
print(features[:2])
print(poly_features[:2,:])
print(labels[:2])
'''
#损失函数
def evaluate_loss(net,data_iter,criterion):
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for x,y in data_iter:
            output = net(x)
            y = y.reshape(-1,1)
            loss = criterion(output,y)
            metric.add (loss.sum(),loss.numel())

    return metric[0] / metric[1]



#写一个把特征和标签合并成dataset的类
class dataset:
    def __init__(self,features,labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)


    def __getitem__(self,idx):
        return self.features[idx,:] , self.labels[idx]

#再写一个按照batch_size来管理dataset的函数

def generate_dataloader(dataset,batch_size):
    length = len(dataset.labels)
    for idx in range(0,length,batch_size):
        end = min(idx + batch_size,length)
        yield dataset.features[idx:end,:] , dataset.labels[idx:end]


#训练函数，画图那部分先不管
def train(train_features,test_features,train_labels,test_labels,num_epochs):
    net = nn.Sequential(nn.Linear(train_features.shape[1],1,bias = False))
    criterion = nn.MSELoss()
    updater = torch.optim.SGD(net.parameters(),lr = 0.01)
    batch_size = min(10,train_labels.shape[0])#?
    train_set = dataset(train_features,train_labels)
    
    test_set = dataset(test_features,test_labels)
    
    animator = d2l.Animator(xlabel="epoch",ylabel = "loss",yscale='log',
    xlim=[1,num_epochs],ylim=[1e-3,1e2],legend=['train','test'])
    for epoch in range(num_epochs):
        test_iter = generate_dataloader(test_set, batch_size)
        train_iter = generate_dataloader(train_set, batch_size)
        #train_utils.train_epoch(net,updater,criterion,train_iter,net.parameters())
        for x,y in train_iter:
            updater.zero_grad()
            output = net(x)
            y = y.reshape(-1,1)
            train_loss = criterion(output,y)
            train_loss.backward()
            updater.step()
        
        if (epoch+1)%20==0 or epoch==0:
            test_iter = generate_dataloader(test_set, batch_size)
            train_iter = generate_dataloader(train_set, batch_size)
            test_loss = evaluate_loss(net,test_iter,criterion)
            train_loss = evaluate_loss(net,train_iter,criterion)
            animator.add(epoch+1,[train_loss,test_loss])
            plt.pause(0.1)  # 暂停一小段时间以更新图形
    print('weight',net[0].weight.data.numpy())#?
    
            #画图
#1.正常的

train(poly_features[:num_train,:4],poly_features[num_train:,:4],
labels[:num_train],labels[num_train:],400)

#2.太少了
'''
train(poly_features[:num_train,:2],poly_features[num_train:,:2],
labels[:num_train],labels[num_train:],400)
'''
#3.太多了
'''
train(poly_features[:num_train,:],poly_features[num_train:,:],
labels[:num_train],labels[num_train:],400)'''
