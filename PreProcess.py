#

import collections
import re

from d2l import torch as d2l

#

#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine()                                :
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# length of text: {len(lines)}')
print(lines[0])
print(lines[10])

#

def tokenize(lines, token = 'word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('error: unknown token: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])

#

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key = lambda x: x[1], reverse=True)
        self.unk = 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token : idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            else:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens(index) for index in indices]

    def token_freqs(self):
        return self._token_freqs





def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

#

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items)[:10])

#

for i in [0, 10]:
    print('text: ' + tokens[i])
    print('indices: ' + vocab[tokens[i]])

#

def load_corpus_time_machine(max_tokens = -1):
    lines = read_time_machine()
    tokens = tokenize(lines)
    vocab = Vocab(tokens)

    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
