import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch.utils import data
from torch.utils.data import DataLoader

scale = 0.01
w1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
w2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
w3 = torch.randn(size=(800, 128)) * scale
b3= torch.zeros(128) 
w4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)

params = [w1, b1, w2, b2, w3, b3, w4, b4]

def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2),
            stride=2)
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=2)
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

loss = nn.CrossEntropyLoss(reduction='none')

def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)

def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)

    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device)

data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i+1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
'''
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
'''

'''
split = []

for i in range(len(devices)):
    size = data.shape[0] // len(devices)
    if i == len(devices) - 1:
        split.append(data[i*size:].to(devices[-1]))
        break
    split.append(data[i*size: (i+1) * size].to(devices[i]))
''' 

'''
print('input :', data)
print('load into :', devices)
print('output :', split)
'''
def split_batch(X, y, devices):
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices), 
    nn.parallel.scatter(y, devices))

def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    ls = [
        loss(lenet(X_shard, device_w),
            y_shard).sum() for X_shard, y_shard, device_w in zip(
                X_shards, y_shards, device_params
            )
    ]

    for l in ls:
        l.backward()

    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])
        
def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4


def train(num_gpus, batch_size, lr):
    resize = 28
    trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(resize,antialias=True)])
    #trans.insert(0,transforms.Resize(resize))
    mnist_train = torchvision.datasets.FashionMNIST(
        root=r'I:\Data\MNIST', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=r'I:\Data\MNIST', train=False, transform=trans, download=True)
    train_iter = DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
    test_iter = DataLoader(mnist_test, batch_size, shuffle = False,
                            num_workers = get_dataloader_workers())  
    #train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs],
        legend = ['train loss','train acc','test acc'])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        animator.add(epoch+1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]
        ) ))

    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.sum()/num_epochs} sec/epoch on {str(devices)}')










train(num_gpus=1, batch_size=256, lr=0.2)