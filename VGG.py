import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torchvision import transforms
from torch.utils import data

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride = 2))
    return nn.Sequential(*layers)
    

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for num_convs, out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
        nn.Dropout(p=0.5), nn.Linear(4096,4096), nn.ReLU(),
        nn.Dropout(p=0.5), nn.Linear(4096,10)
    )




#参数


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, "output shape:\t",X.shape)


ratio = 4
small_conv_arch = [(pair[0], pair[1]//ratio) for pair in conv_arch]
print(small_conv_arch)
small_net = vgg(small_conv_arch)

lr,num_epochs,batch_size = 0.05, 10, 128

resize = 224
trans = transforms.Compose([transforms.ToTensor(),transforms.Resize(resize)])
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
