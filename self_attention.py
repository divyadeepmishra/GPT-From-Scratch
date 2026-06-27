import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self,x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2,-1)  #Every token compared with every token.
        scores = scores / (K.size(-1) ** 0.5) #Prevents exploding values

        mask = torch.trill(
            torch.ones(
                scores.size(-2),
                scores.size(-1),
                device=x.device
            )
        )

        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention  = F.softmax(scores, dim = -1) #Attention weights

        output = attention @ V #context-aware embeddings

        return output
