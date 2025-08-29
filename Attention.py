#

import torch
from d2l import torch as d2l
from torch import nn

#

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                    cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(
        num_rows, num_clos, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrtix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_row-1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles)
    fig.colorbar(pcm, ax=axes, shrink=0.6)

#

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')

#

n_train = 50
x_train, _ = torch.sort(torch.rand(n_train)*5)

def f(x):
    return 2 * torch.sin(x) + x ** 8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train, ))

x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)
n_test

#

def plot_kernel_reg(y_hat):
    d2l.plot(
        x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], 
        xlim=[0, 5], ylim=[-1, 5]
    )
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

#

y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)

#
# (n_test) -> (n_test, n_train)
X_repeat = x_test.repeat_interleave(n_train).reshape(-1, n_train)
#                                    (n_test, n_train)  - ( (n_train) => (n_test, n_train) )
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)

y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

#

d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
        xlabel= 'Sorted training inputs',
        ylabel= 'Sorted testing inputs' )

#

X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
torch.bmm(X, Y).shape

#

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
    def forward(self, queries, keys, values):

        queries = queries.repeat_interleave(keys.shape[1]).reshape(-1, keys.shape[1])
        self.attention_weights = nn.functional.softmax(
            -((queries - keys)* self.w)**2 / 2, dim=1
        )
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1).reshape(-1))

#

X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))
keys = X_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = Y_tile[(1-torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

#

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch+1}, loss {l.sum():.6f}')
    animator.add(epoch+1, float(l.sum()))

#

keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

#

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')