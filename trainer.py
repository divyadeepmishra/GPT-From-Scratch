import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import GPTConfig
from dataset import GPTDataset
from model import MiniGPT

class Trainer:
    def __init__(self, config):
        self.config = config
        torch.manual_seed(config.seed)

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(config.seed)

        self.device = torch.device(config.device)
        self.model = MiniGPT(config).to(self.device)
        self.train_dataset = GPTDataset("data/processed/train.bin", config.block_size)
        self.val_dataset = GPTDataset("data/processed/val.bin", config.block_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

    def train(self):
        print(f"\nTraining on {self.device}\n")

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0

            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                B, T, C = logits.shape
                loss = self.criterion(logits.view(B * T, C), y.view(B * T))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            val_loss = self.validate()
            print(f"Epoch [{epoch+1}/{self.config.epochs}] | "f"Train Loss: {avg_train_loss:.4f} | "f"Validation Loss: {val_loss:.4f}")

        torch.save(self.model.state_dict(), "minigpt.pt")
        print("\nTraining Complete!")
        print("Model saved as minigpt.pt")

    @torch.no_grad()
    def validate(self):

        self.model.eval()
        total_loss = 0.0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            B, T, C = logits.shape
            loss = self.criterion(logits.view(B * T, C), y.view(B * T))
            total_loss += loss.item()

        return total_loss / len(self.val_loader)


if __name__ == "__main__":
    config = GPTConfig()
    trainer = Trainer(config)
    trainer.train()
