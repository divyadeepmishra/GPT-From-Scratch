import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.embed_dim = config.embed_dim
        self.head_dim = (self.embed_dim // self.n_heads)

        assert (self.head_dim * self.n_heads == self.embed_dim)

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        scores = (Q @ K.transpose(-2,-1)) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))

        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(scores, dim=-1)

        attention = self.dropout(attention)

        output = attention @ V

        output = output.transpose(1, 2).contiguous()

        output = output.view(B, T, C)

        output = self.projection(output)

        return output
