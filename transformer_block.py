import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = LayerNorm(config)
        self.norm2 = LayerNorm(config)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
