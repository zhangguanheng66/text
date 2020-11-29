import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Transformer, Embedding


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AnnotatedTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab,
                 N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(AnnotatedTransformer, self).__init__()
        self.transformer = Transformer(d_model=d_model, nhead=h,
                                       num_encoder_layers=N, num_decoder_layers=N,
                                       dim_feedforward=d_ff, dropout=dropout)
        self.d_model = d_model
        self.src_embed = Embedding(src_vocab, d_model)
        self.src_pos_embed = PositionalEncoding(d_model, dropout)
        self.tgt_embed = Embedding(tgt_vocab, d_model)
        self.tgt_pos_embed = PositionalEncoding(d_model, dropout)
        self.generator = Generator(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.src_pos_embed(self.src_embed(src) * math.sqrt(self.d_model))
        tgt = self.tgt_pos_embed(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        # print(src.is_cuda, tgt.is_cuda, src_mask.is_cuda, tgt_mask.is_cuda)
        out = self.transformer(src, tgt, src_mask, tgt_mask)
        return self.generator(out)
