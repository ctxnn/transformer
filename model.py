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
            pe[:,0::2] = torch.sin(pos*div) # sin(position * (10000 ** (2i / d_model))
            # to odd indices
            pe[:,1::2] = torch.cos(pos*div) # cos(position * (10000 ** (2i / d_model))
            # Add a batch dimension to the positional encoding
            pe = pe.unsqueeze(0) # (1, seq_len, d_model)
            # registering the pe matrix to buffer cause we dont want any learning on it, it will stay the same for every batch
            self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1], :].requires_grad_(False)) # (batch, seq_len, d_model)
        return self.dropout(x)

# layer normalization
class LayerNormalization(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e4):
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
class FeedForwardBlock(nn.Module):
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
        self.w_o = nn.Linear(d_model,d_model) #actually its (d_k*h,d_model) but d_k*h is just d_model

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
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)
        #calculating attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        #concatinate/combine the multiple heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

# skip/residual connection
class ResidualConnection(nn.Module):

        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))


# encoder block (cause there can be N no of encoders)
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# encoder containing all the no of encoder blocks
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__() # Call super().__init__() FIRST
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask) # updating the x
        return self.norm(x)

# decoder block (cause there can be N no of decoders)
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

# Decoder contains all the n no of decoder blocks
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__() # Call super().__init__() FIRST
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# final linear layer (turns the pos into vocab)
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncodings, tgt_pos: PositionalEncodings, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

# now that all the components of transformers are there, writing a function to build a transformer from these functions
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:

    #word embeddings for source language and target language( we are building this transformer for machine translation)
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #positional encodings for both
    src_pos = PositionalEncodings(src_seq_len, d_model, dropout)
    tgt_pos = PositionalEncodings(tgt_seq_len, d_model, dropout)

    # encoder
    encoder_blocks = [] #empty array to store all the blocks
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(h, d_model, dropout)
        encoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # decoder
    decoder_blocks = [] #empty array to store all the blocks
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(h, d_model, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(h, d_model, dropout)
        decoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    # Creating the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    #projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
