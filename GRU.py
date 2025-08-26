import math
import torch 
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import re
import collections

def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps-1):]
    # -1 因为需要考虑标签
    num_subseqs = (len(corpus)-1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    random.shuffle(initial_indices)    

    def data(pos):
        return corpus[pos: pos+num_steps]
    
    num_batches = num_subseqs // batch_size

    for i in range(0, num_batches * batch_size, batch_size):

        initial_indices_per_batch = initial_indices[i, i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps-1)
    num_tokens = ((len(corpus) - offset - 1) // num_steps) * num_steps
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + num_tokens + 1])
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps*num_batches, num_steps):
        X = Xs[:, i:i+num_steps]
        Y = Ys[:, i:i+num_steps]
        yield X, Y

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine()                                :
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def load_corpus_time_machine(max_tokens = -1):
    lines = read_time_machine()
    tokens = d2l.tokenize(lines, token='char')
    vocab = d2l.Vocab(tokens)

    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab

class SeqDataloader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential

        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataloader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


batch_size, num_steps = 32, 35
train_iter, vocab  = load_data_time_machine(batch_size, num_steps)

#

def get_params(vocab_size, num_hiddens, device):

    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    
    def three():
        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device)
        )

    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for p in params:
        p.requires_grad_(True)
    return params

#

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

#

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q= params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

#

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens,
         device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return init_gru_state(batch_size, self.num_hiddens, device)



def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


#

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for p in params:
            p.grad[:] *= theta / norm

#

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)

    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(X.shape[0], device)
        else:
            if isinstance(net, nn.Module):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        
        y = Y.T.reshape(-1) # (batch_size, num_steps) -> (batch_size * num_steps)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state) # y_hat: (num_steps * batch_size, vocab_size)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return  math.exp(metric[0]/metric[1]), metric[1] / timer.stop()

#

def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator =d2l.Animator(
        xlabel='epoch', ylabel='perplexity',
        legend='train', xlim=[10, num_epochs]
    )

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.parameters(), lr, batch_size) 
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    for epoch in num_epochs:
        ppl, speed = train_epoch_ch8(
                net, train_iter, loss, updater, device, use_random_iter)
        if (epoch+1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch+1, [ppl])
    
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))



vocab_size, num_hiddens, device = len(vocab), 256, d2l.d2l.try_gpu()

num_epochs, lr = 500, 1
model = RNNModelScratch(
    vocab_size, num_hiddens, device, get_params, init_gru_state, gru)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)

#

class RNNModel(nn.Module):
    
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens*2, self.vocab_size)
        

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((
                self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens),
                device=device)
        
        else:
            return (torch.zeros((
                self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens),
                device=device),
                torch.zeros((
                self.num_directions*self.rnn.num_layers, batch_size, self.num_hiddens),
                device=device))




num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = RNNModel(gru_layer, vocab_size)
train_ch8(model ,train_iter, vocab, lr, num_epochs, device)


#

def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    
    def three():
        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device)
        )

    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for p in params:
        p.requires_grad_(True)
    return params

#

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

#


def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.sigmoid((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
    
#

vocab_size, num_hiddens, device = len(vocab), 256, d2l.d2l.try_gpu()

num_epochs, lr = 500, 1
model = RNNModelScratch(
    vocab_size, num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)

#

num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = RNNModel(lstm_layer, vocab_size)
model = model.to(device)
train_ch8(model ,train_iter, vocab, lr, num_epochs, device)