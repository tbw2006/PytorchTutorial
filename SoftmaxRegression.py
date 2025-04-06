import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l


def get_data_fashion_mnist(batch_size = 32,resize = None):

    trans = transforms.Compose([transforms.ToTensor()])
    if resize:
        trans.insert(0,transforms.Resize(resize))
    train_set = torchvision.datasets.FashionMNIST(
        root=r'I:\Data\MNIST',train=True,
        transform= trans,download=True
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=r'I:\Data\MNIST',train=False,
        transform=trans,download=True
    )
    print(len(train_set))
    print(len(test_set))
    print(test_set[0][0].shape)

    
    return (data.DataLoader(dataset=train_set,batch_size=batch_size,
                                shuffle=True),
            data.DataLoader(dataset=test_set,batch_size=batch_size,
                                shuffle=False)       )
    timer = d2l.Timer()                 #这里可以看读取速度
    for x,y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')     



batch_size = 32
train_iter ,test_iter = get_data_fashion_mnist(batch_size)

#初始化
input_size = 784
output_size = 10
w = torch.normal(0,0.01,size = (input_size,output_size),requires_grad=True)
b = torch.zeros(output_size,requires_grad=True)


def softmax(x):
    exp_x = torch.exp(x)
    partition = exp_x.sum(dim=1,keepdim=True)
    return exp_x / partition
#测试一下
x = torch.normal(0,0.1,(2,5))
x = softmax(x)
print(x)
print(x.sum(1))


def net(x):
    return softmax(torch.matmul(x.reshape(-1,w.shape[0]),w)+b)

def cross_entropy(y_hat,y):
    return -1 * torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    predict = torch.argmax(y_hat,dim = 1)
    correct = (y==predict).sum().item()
    return correct 

def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    total = 0
    correct = 0
    for x,y in data_iter:
        y_hat = net(x)
        total += len(x)
        correct+=accuracy(y_hat,y)

    return correct/total

#print(evaluate_accuracy(net,test_iter))测试
def train_epoch(net,updater,criterion,train_iter):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)         #0:loss,1:correct,2:total
    for x,y in  train_iter:
        y_hat = net(x)
        loss = criterion(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            loss.backward()
            updater.step()
            metric.add(loss.sum().item(),accuracy(y_hat,y),len(y))

        else:
            loss.sum().backward()#只有标量才能反向传播
            updater([w,b])#what in ()
            metric.add(loss.sum().item(),accuracy(y_hat,y),len(y))

    return metric[0]/metric[2] , metric[1]/ metric[2]


def sgd(params,lr = 0.03,batch_size = 32):
    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param -= lr *  param.grad / batch_size
                param.grad.zero_()

net = net
updater = sgd
num_epochs = 10
lr = 0.03
criterion = cross_entropy
def train(net,updater,train_iter,test_iter,num_epochs):
    for epoch in range(num_epochs):
        loss , train_accuracy = train_epoch(net,updater,criterion,train_iter)
        test_accuracy = evaluate_accuracy(net,test_iter)
        print(f'epoch: {epoch+1} , loss: {loss:.4f}' )
        print(f'train_accuracy: {train_accuracy:.4f} , test_accuracy: {test_accuracy}')

train(net,updater,train_iter,test_iter,num_epochs)
