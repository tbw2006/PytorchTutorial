import torch
from torch import nn
from d2l import torch as d2l
from torch.onnx.symbolic_opset18 import col2im
import torchvision
from torchvision import transforms
from torch.utils import data

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(), nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(), nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU()
    )

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(3, stride=2), nn.Dropout(p=0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
    
    
)


X = torch.randn(size = (1,1,224,224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, "output shape:\t", X.shape)


lr, num_epochs, batch_size = 0.1, 10, 128
resize = 224
trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(resize,antialias=True)])
#trans.insert(0,transforms.Resize(resize))
mnist_train = torchvision.datasets.FashionMNIST(
    root=r'I:\Data\MNIST', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root=r'I:\Data\MNIST', train=False, transform=trans, download=True)



def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
test_iter = data.DataLoader(mnist_test, batch_size, shuffle = False,
                            num_workers = get_dataloader_workers())  
d2l.train_ch6(net,train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

