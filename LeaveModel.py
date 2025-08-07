from PIL import Image
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data
import pandas as pd
import os
from d2l import torch as d2l

from Basic import y_clone


class CustomImageDataset(data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None, has_labels=True):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels
        if has_labels:
            
            self.classes = sorted(self.img_labels.iloc[:, 1].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            print(f'Found {len(self.classes)} classes')

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename = self.img_labels.iloc[idx, 0].split('/')[-1].split('\\')[-1]
        img_name = os.path.join(self.img_dir, img_filename)
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        if self.has_labels:
            label_str = self.img_labels.iloc[idx,1]
            label = self.class_to_idx[label_str]
            return image, label
        else:
            return image, torch.tensor([-1],dtype=torch.long)
        

transform = transforms.Compose([transforms.ToTensor()])
train_csv_file = r'D:\tbw\Pytorch\Competition\leaves-data\train.csv'
test_csv_file = r'D:\tbw\Pytorch\Competition\leaves-data\test.csv'
img_dir = r'D:\tbw\Pytorch\Competition\leaves-data\images'
batch_size=32
train_data = CustomImageDataset(train_csv_file, img_dir, transform, has_labels=True)
train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_data = CustomImageDataset(test_csv_file, img_dir, transform, has_labels=False)
test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)


for X,y in train_loader:
    print(X.shape)
    #print(y)
    break

net = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096,176)
)

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on: ', device)
    net = net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr)
    animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],
                            legend = ['train loss','train acc'])
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
        #test_acc = d2l.evaluate_accuracy_gpu(net,test_iter)
        #animator.add(epoch + 1, (None, None, test_acc) )
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')
    print((f'{metric[2] * num_epochs / timer.sum():.1f}  examples/sec'),
            f'on  {str(device)}')


lr, num_epochs = 0.01, 10
train(net,train_loader, test_loader, num_epochs, lr, d2l.try_gpu())    


def get_idx_to_class(csv_file):

        img_labels = pd.read_csv(csv_file)
        classes = sorted(img_labels.iloc[:, 1].unique())
        idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
        print(f'generate {len(self.classes)} indices to class')
        return idx_to_class

def generate_csv(net, test_loader, idx_to_class, device):
    net.eval()
    all_filenames = []
    all_predictions = []
    for X, _ in test_loader:
        with torch.no_grad():
            X = X.to(device)
            Y_hat = net(X)
            Y_hat = torch.argmax(Y_hat, dim=1)
            Y_hat = Y_hat.cpu().numpy()
            batch_filenames = [f'image/{i}.ipg' for i in range(len(Y_hat))]
            all_filenames.extend(batch_filenames)
            all_predictions.extend(Y_hat)
    pred_classes = [idx_to_class[pred] for pred in all_predictions]
    submission = pd.DataFrame({'image': all_filenames, 'label': pred_classes})  ###

    submission.to_csv(r'D:\tbw\Pytorch\Competition\leaves-data\submissionsubmission.csv',
        index = False)
    print('submisson file has been created succsessfully')

    return submission

idx_to_class = get_idx_to_class(train_csv_file)
generate_csv(net, test_loader, idx_to_class, d2l.try_gpu())


