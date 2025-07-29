import torch
from torch import nn
from d2l import torch as d2l
from torchvision import transforms
from torch.utils import data
import torchvision

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096,10)
)


X = torch.rand(size=(1,1,224,224),dtype=torch.float32)

for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)


batch_size = 128
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
lr, num_epochs = 0.01, 10
d2l.train_ch6(net,train_iter, test_iter, num_epochs, lr, d2l.try_gpu())