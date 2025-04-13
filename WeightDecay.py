from turtle import up
import torch
from torch import nn
from d2l import  torch as d2l
import train_utils
import matplotlib.pyplot as plt
plt.ion()  # 打开交互模式
num_train,num_test,num_inputs,batch_size = 20,100,200,5
#我需要生成一些数据


#初始化w,b
true_w = torch.ones((num_inputs,1)) *0.01
true_b = torch.ones(1) * 0.05
w = torch.normal(0,0.1,(num_inputs,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

train_x,train_y = train_utils.synthetic_data(true_w,true_b,num_train)
test_x,test_y = train_utils.synthetic_data(true_w,true_b,num_test)

train_iter = train_utils.data_iter(train_x,train_y,batch_size)
test_iter = train_utils.data_iter(test_x,test_y,batch_size)

def return_weight_decay(w):
    return torch.sum(w**2) / 2

def weight_decay_criterion():
    def mixed_criterion(lambd,y,y_hat,criterion = train_utils.cross_entropy):
        
        return criterion(y,y_hat) + lambd * return_weight_decay(w)
    return mixed_criterion




def weight_decay_train(lambd):
    criterion = train_utils.weight_decay_criterion()
    
    net = train_utils.linreg
    num_epochs ,lr= 100,0.003
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',
    xlim=[5,num_epochs],legend=['train','test'])
    params = [w,b]
    updater = train_utils.sgd()
    for epoch in range(num_epochs):
        train_iter = train_utils.data_iter(train_x,train_y,batch_size)
        test_iter = train_utils.data_iter(test_x,test_y,batch_size)
        for x,y in train_iter:
            
           
            y_hat = net(x,w,b)
            train_loss = criterion(lambd,w,y,y_hat,train_utils.squared_loss).sum()
            train_loss.backward()

            

            updater(params,lr,batch_size)    

            
    
            
        for x,y in test_iter:
            with torch.no_grad():
                y_hat = net(x,w,b)
                test_loss = criterion(lambd,w,y,y_hat,train_utils.squared_loss).sum()
        if (epoch+1)%5==0:
            animator.add(epoch+1,[train_loss.item(),test_loss.item()])
            plt.pause(0.1)  # 添加短暂暂停来刷新图形
            print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}")
    print('w\'s L2 is ',return_weight_decay(w).item() )



weight_decay_train(0)
plt.show(block=True)  # 添加这一行
    

