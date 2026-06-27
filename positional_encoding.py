import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, block_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(block_size, embed_dim)

    def forward(self, token_embeddings):
        sequence_length = token_embeddings.size(1)
        positions = torch.arange(sequence_length, device = token_embeddings.device)
        position_embeddings = self.embedding(positions)
        return token_embeddings + position_embeddings
