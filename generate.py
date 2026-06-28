import torch
import torch.nn.functional as F
from config import GPTConfig
from model import MiniGPT
from tokenizer import Tokenizer
from pathlib import Path

config = GPTConfig()
device = torch.device(config.device)

#Initialize the model and tokenizer
model = MiniGPT(config).to(device)
tokenizer = Tokenizer()

model.load_state_dict(torch.load("checkpoints/minigpt.pt", map_location=device))

model.eval()

prompt = input("Prompt: ")
tokens = tokenizer.encode(prompt)
tokens = torch.tensor([tokens], dtype=torch.long, device=device)

with torch.no_grad():
    for _ in range(100):
        context = tokens[:, -config.block_size:]
        logits = model(context)
        logits = logits[:, -1, :]
        logits = logits / config.temperature
        values, indices = torch.topk(logits, config.top_k)

        probs = F.softmax(values, dim=-1)
        next_index = torch.multinomial(probs, num_samples=1)
        next_token = indices.gather(-1, next_index)
        tokens = torch.cat([tokens, next_token], dim=1)
output_text = tokenizer.decode(tokens[0].tolist())

print()
print(output_text)
Path("outputs").mkdir(exist_ok=True)
with open("outputs/sample.txt", "w", encoding="utf-8") as file:
    file.write(output_text)
