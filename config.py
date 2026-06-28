import torch
from dataclasses import dataclass

@dataclass
class GPTConfig:

    vocab_size = 50257  #Vocabulary
    block_size = 128    #Max content length

    #Transformer
    n_layers = 6
    n_heads = 8
    embed_dim = 512

    mlp_ratio = 4   # feed forward
    dropout = 0.1   # regularization

    #Training
    batch_size = 32
    learning_rate = 3e-4
    epochs = 5
    seed = 42

    #Generation
    temperature = 1.0
    top_k = 50

    #device
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
