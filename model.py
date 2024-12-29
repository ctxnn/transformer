
import torch
import torch.nn as nn
import math

# input embeddings
class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    # (batch, seq_len) --> (batch, seq_len, d_model)
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model) # given in the model

# positional encodings
class PositionalEncodings(nn.Module):
    def __init__(self, seq_length:int, d_model:int, dropout:float):
            super().__init__()
            self.seq_length = seq_length
            self.d_model = d_model
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(seq_length, d_model)
            # pos matrix of size(seq_length,1)(numerator term of the formula)
            pos = torch.arange(0,seq_length, dtype=torch.float).unsqueeze(1)
            # div matrix of size(d_model, 1)(denominator term of the formula)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            # to even indices
            pe[0::2] = torch.sin(pos*div) # sin(position * (10000 ** (2i / d_model))
            # to odd indices
            pe[1::2] = torch.cos(pos*div) # cos(position * (10000 ** (2i / d_model))
            # Add a batch dimension to the positional encoding
            pe = pe.unsqueeze(0) # (1, seq_len, d_model)
            # registering the pe matrix to buffer cause we dont want any learning on it, it will stay the same for every batch
            self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, x[1], :].requires_grad_(False)) # (batch, seq_len, d_model)
        return self.dropout(x)

# layer normalization
class LinearNormalization(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e**4):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.ones(1)) # added

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# feed forward block
class FeedForwardBlock(nn.Module)
    def __init__(self, d_model:int, d_ff:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self. dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model) # acc to paper the size for the ff layer goes from dmodel -> dff -> dmodel

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# multi attention block
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, h:int, d_model:int, dropout:float):
        super().__init__()
        self.h = h # no of heads
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // h # Dimension of vector seen by each head
        # d_model is divisible by h
        assert d_model % h == 0, "d_model not divisible by h"
        # W_q
        self.w_q = nn.Linear(d_model,d_model)
        # W_k
        self.w_k = nn.Linear(d_model,d_model)
        # W_v
        self.w_v = nn.Linear(d_model,d_model)
        #W_o
        self.w_0 = nn.Linear(d_model,d_model) #actually its (d_k*h,d_model) but d_k*h is just d_model

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        # using the formula (q*k^t/sqrt(d_k))
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            # low value = -inf , to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q)# (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)# (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)# (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        #calculating attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        #concatinate/combine the multiple heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
