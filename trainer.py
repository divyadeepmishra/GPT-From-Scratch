import torch
import torch.nn as nn
import torch.optim as optim
from config import GPTConfig
from model import MiniGPT

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.model = MiniGPT(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)

    def train_step(self, x, y):
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        B, T, C = logits.shape
        loss = self.criterion(logits.view(B*T, C), y.view(B*T))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
