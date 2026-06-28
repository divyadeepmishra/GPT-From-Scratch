import torch.nn as nn
from embeddings import TokenEmbedding
from positional_encoding import PositionalEncoding
from transformer_block import TransformerBlock
from layer_norm import LayerNorm

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = TokenEmbedding(config.vocab_size, config.embed_dim)
        self.position_embedding = PositionalEncoding(config.block_size, config.embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = LayerNorm(config)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        x = self.position_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        logits = self.lm_head(x)
        return logits
