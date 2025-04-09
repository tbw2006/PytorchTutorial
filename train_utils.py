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
    #print(len(train_set))
    #print(len(test_set))
    #print(test_set[0][0].shape)

    
    return (data.DataLoader(dataset=train_set,batch_size=batch_size,
                                shuffle=True),
            data.DataLoader(dataset=test_set,batch_size=batch_size,
                                shuffle=False)       )
    '''timer = d2l.Timer()                 #这里可以看读取速度
    for x,y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')     '''


def softmax(x):
    x_max = x.max(dim=1,keepdim = True)[0]
    exp_x = torch.exp(x-x_max)
    partition = exp_x.sum(dim=1,keepdim=True)
    return exp_x / partition
#测试一下
'''x = torch.normal(0,0.1,(2,5))
x = softmax(x)
print(x)
print(x.sum(1))'''


def accuracy(y_hat,y):
    predict = torch.argmax(y_hat,dim = 1)
    correct = (y==predict).sum().item()
    return correct 


def cross_entropy(y_hat,y):
    return -1 * torch.log(y_hat[range(len(y_hat)),y] +1e-10)
#   +1e-10防止log0的出现-> 负无穷


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


def train_epoch(net,updater,criterion,train_iter,params):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)         #0:loss,1:correct,2:total
    for x,y in  train_iter:
        y_hat = net(x)
        loss = criterion(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,max_norm=10.0)
            updater.step()
            metric.add(loss.sum().item(),accuracy(y_hat,y),len(y))

        else:
            loss.sum().backward()#只有标量才能反向传播
            updater(params)
            metric.add(loss.sum().item(),accuracy(y_hat,y),len(y))

    return metric[0]/metric[2] , metric[1]/ metric[2]



def sgd(params,lr = 0.03,batch_size = 32):
    def updater(_):
        with torch.no_grad():
            for param in params:
                if param.grad is not None:
                    
                    param -= lr *  param.grad / batch_size
                    param.grad.zero_()
    return updater




def train(net,train_iter,test_iter,
updater=None,num_epochs=10,params=None,lr = 0.03,criterion=cross_entropy):
    if updater is None and params is not None:
        updater = sgd(params=params,lr=lr)
    for epoch in range(num_epochs):
        loss , train_accuracy = train_epoch(net,updater,criterion,train_iter,params)
        test_accuracy = evaluate_accuracy(net,test_iter)
        if params is not None:
            print(f"w1范围: min={params[0].min().item():.4f}, max={params[0].max().item():.4f}")
            print(f"w2范围: min={params[2].min().item():.4f}, max={params[2].max().item():.4f}")
        print(f'epoch: {epoch+1} , loss: {loss:.4f}' )
        print(f'train_accuracy: {train_accuracy:.4f} , test_accuracy: {test_accuracy}')