#

import random
import torch
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine()                                :
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

tokens = d2l.tokenize(read_time_machine())

corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[0:10]

#

freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(
    freqs, xlabel='token: x', ylabel='frequency: n(x)',
     xscale='log', yscale='log')

#

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[0:10]

#

trigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[0:10]

#

bigram_freqs = [
    freq for token, freq in bigram_vocab.token_freqs
]
trigram_freqs = [
    freq for token, freq in trigram_vocab.token_freqs
]
d2l.plot(
    [freqs, bigram_freqs, trigram_freqs],
    xlabel='token: x', ylabel='frequency: n(x)',
    xscale='log', yscale='log',
    legend=['unigram', 'bigram', 'trigram']
)

#

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


#

my_seq = list(range(35))
for X, y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\ny: ', y)

#

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

#

for X, y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\ny: ', y)

#

def load_corpus_time_machine(max_tokens = -1):
    lines = read_time_machine()
    tokens = d2l.tokenize(lines)
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

        self.corpus, self.vacab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

#

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataloader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab




