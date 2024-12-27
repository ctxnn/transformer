
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
