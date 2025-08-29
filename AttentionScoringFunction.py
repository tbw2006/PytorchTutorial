#

import math
import torch
from torch import nn
from d2l import torch as d2l

#

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

#

masked_softmax(torch.rand((2, 2, 4)), torch.tensor([2, 3]))

#

masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))

#

class AdditiveAttention(nn.Module):

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries: (batch_size, num_queries, hidden_size) -> (batch_size, num_queries, 1, num_hiddens)
        # keys: (batch_size, num_keys, hidden_size) -> (batch_size, 1, num_keys, num_hiddens)
        # broadcast: features: (batch_size, num_queries, num_keys, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

#

queries, keys = torch.normal(0, 1, (2, 1, 10)), torch.ones((2, 10, 2))

values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=10, num_hiddens=8, dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)

#

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
        xlabel='Keys', ylabel='Queries')

#

class DotProductAttention(nn.Module):

    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

#

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)

#

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

 
