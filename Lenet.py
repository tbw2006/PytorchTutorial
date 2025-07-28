from random import shuffle
import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torchvision import transforms
from torch.utils import data


class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1,1,28,28)

net = nn.Sequential(
    Reshape(), nn.Conv2d(1,6,kernel_size=5, padding = 2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride = 2), nn.Conv2d(6,16,kernel_size=5),
    nn.Sigmoid(), nn.AvgPool2d(kernel_size=2,stride=2), nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(), nn.Linear(120,84),
    nn.Sigmoid(), nn.Linear(84,10)
)


X = torch.rand(size = (1,1,28,28),dtype=torch.float32)

for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


def evaluate_accuracy_gpu(net, data_iter, device = None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    for X,y in data_iter:
        if isinstance(X,list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X),y),y.numel())
    return metric[0] / metric[1]




def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on: ', device)
    net = net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr)
    animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],
                            legend = ['train loss','train acc','test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X,y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], d2l.accuracy(y_hat,y),X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i+1) % (num_batches // 5) == 0 or i == num_batches + 1:
                #print(f"X is on: {X.device}")  # 应该输出 cuda:0
                #print(f"y is on: {y.device}")  # 应该输出 cuda:0
                #print(f"Model is on: {next(net.parameters()).device}")  # 应该输出 cuda:0
                
                animator.add(epoch + (i+1) / num_batches,
                            ( train_l,train_acc,None))
        test_acc = d2l.evaluate_accuracy_gpu(net,test_iter)
        animator.add(epoch + 1, (None, None, test_acc) )
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}', f'test acc {test_acc:.3f}')
    print((f'{metric[2] * num_epochs / timer.sum():.1f}  examples/sec'),
            f'on  {str(device)}')


# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root=r'I:\Data\MNIST', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root=r'I:\Data\MNIST', train=False, transform=trans, download=True)


batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
test_iter = data.DataLoader(mnist_test, batch_size, shuffle = False,
                            num_workers = get_dataloader_workers())                             


lr, num_epochs = 0.9, 10
train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())                