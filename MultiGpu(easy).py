import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch.utils import data
from torch.utils.data import DataLoader


def resnet18(num_classes, in_channels=1):
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if not first_block and i == 0:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1,1)), 
                nn.Flatten(), nn.Linear(512, 10))
    return net

def train(net, num_gpus, batch_size, lr):
    resize = 28
    trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(resize,antialias=True)])
    #trans.insert(0,transforms.Resize(resize))
    mnist_train = torchvision.datasets.FashionMNIST(
        root=r'I:\Data\MNIST', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=r'I:\Data\MNIST', train=False, transform=trans, download=True)
    train_iter = DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=4)
    test_iter = DataLoader(mnist_test, batch_size, shuffle = False,
                            num_workers =4)  
    #train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on: ', str(devices))
    net = nn.DataParallel(net, device_ids=devices)
    
    loss = nn.CrossEntropyLoss()
    num_epochs = 10
    optimizer = torch.optim.SGD(net.parameters(),lr)
    animator = d2l.Animator(xlabel='epoch',ylabel='test acc',xlim=[1,num_epochs],
                            )
    timer, num_batches = d2l.Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X,y in train_iter:
            optimizer.zero_grad()
            X,y = X.to(devices[0]), y.to(devices[0])
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        timer.stop()
            
        
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net,test_iter)) )
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.sum()/num_epochs:.2f} sec/epoch',
        f'on {str(devices)}')

net = resnet18(10, 1)
train(net, 1, 256, 0.1)