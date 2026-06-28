from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):

    def __init__(self, file_path, block_size):
        self.block_size = block_size
        self.stride = block_size
        self.tokens = np.fromfile(Path(file_path), dtype=np.uint16)

    def __len__(self):
         return (len(self.tokens) - self.block_size) // self.stride

    def __getitem__(self, idx):
         idx = idx * self.stride
         x = self.tokens[idx : idx + self.block_size]
         y = self.tokens[idx + 1 : idx + self.block_size + 1]
         return (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
