import hashlib

import os
import tarfile
import zipfile
from d2l.torch import DATA_URL
from pandas.io import feather_format
import requests

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from torchvision.datasets.dtd import download_and_extract_archive

from train_utils import weight_decay_criterion
# 在文件顶部添加
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg' 等其他支持的后端
import matplotlib.pyplot as plt
plt.ion()  # 开启交互模式


DATA_HUB = dict()
DATA_URL = 'https://d2l-data.s3-accelerate.amazonaws.com/'

def download(name,cache_dir=os.path.join('..','data')):
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}."
    url , sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir,exist_ok=True)
    fname = os.path.join(cache_dir,url.split('/')[-1])
    if not os.path.exists(fname):               #下载
        print(f'downloading....')
        r = requests.get(url,stream= True)
        with open(fname,'wb')as f:
            for chunk in r.iter_content(chunk_size = 8192):
                f.write(chunk)

    sha1 = hashlib.sha1()
    with open(fname, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    if sha1.hexdigest() != sha1_hash:
        print(f'warning: file {fname} is not good')


    return fname








DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce'
)

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
)

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

print(train_data.shape)
print(test_data.shape)


print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])    #数据部分打印

all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda  x:((x-x.mean())/x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)  


all_features = pd.get_dummies(all_features,dummy_na=True)
print(all_features.shape)
all_features = all_features * 1
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,dtype = torch.float32)
test_features = torch.tensor(all_features[n_train:].values,dtype = torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1),
        dtype = torch.float32)
#print(train_labels.shape)
# 在数据加载后添加
print(f"Original data shapes:")
print(f"  train_features: {train_features.shape}")
print(f"  train_labels: {train_labels.shape}")

criterion = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

def log_rmse(net,features,labels,criterion = criterion):
    clipped_preds  = torch.clamp(net(features),1,float('inf'))
    rmse = torch.sqrt(criterion(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()

def train(net,train_features,train_labels,test_features,test_labels,learning_rate,
            weight_decay,epochs,batch_size):
    train_ls,test_ls = [],[]
    train_iter = d2l.load_array((train_features,train_labels),batch_size)
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,
    weight_decay = weight_decay)
    for epoch in range(epochs):
        for x,y in train_iter:
            optimizer.zero_grad()
            y_hat = net(x)
            loss = criterion(y,y_hat)
            loss.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))

    return train_ls,test_ls

def get_k_fold_data(k,i,x,y):
    assert k > 1
    fold_size = x.shape[0] // k
    x_train , y_train = None,None
    for j in range(k):
        idx = slice(j * fold_size,(j + 1)*fold_size)
        x_part , y_part = x[idx,:] , y[idx]
        if j == i:
            x_valid , y_valid = x_part , y_part
        elif x_train is None:
            x_train , y_train = x_part , y_part
        else:
            x_train = torch.cat([x_train,x_part],0)
            y_train = torch.cat([y_train,y_part],0)

    return x_train , y_train , x_valid, y_valid

def k_fold(k,learning_rate,weight_decay,num_epochs,batch_size):
    train_l_sum , valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k,i,train_features,train_labels)
        x_train, y_train, x_valid, y_valid = data
        print(f"Fold {i+1}:")
        print(f"  Train features: {x_train.shape}, labels: {y_train.shape}")
        print(f"  Valid features: {x_valid.shape}, labels: {y_valid.shape}")
        net = get_net()
        train_ls, valid_ls = train(net,*data,learning_rate,
                                    weight_decay,num_epochs,batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1,num_epochs + 1)),[train_ls,valid_ls],
            xlabel = 'epoch', ylabel = 'rmse',xlim = [1,num_epochs],
            legend = ['train','valid'],yscale = 'log')
            plt.draw()
            plt.pause(0.1)
            plt.savefig(f'fold_{i+1}_training_curve.png')
            plt.close()  # 关闭图形窗口
        print(f'fold{i+1}, train rmse {float(train_ls[-1])},'
        f'valid log rmse {float(valid_ls[-1]):f}')

    return train_l_sum / k ,valid_l_sum / k


k, num_epochs , learning_rate, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k,learning_rate,weight_decay,num_epochs,batch_size)

print(f'{k}_fold : average train log rmse: {float(train_l):f}',
    f'average valid log rmse: {float(valid_l):f}')



