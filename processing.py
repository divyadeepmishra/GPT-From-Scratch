import json
from pathlib import Path
import numpy as np
from tokenizer import Tokenizer

RAW_PATH = Path("data/raw/corpus.txt")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED_DIR / "train.bin"
VAL_PATH = PROCESSED_DIR / "val.bin"
META_PATH = PROCESSED_DIR / "metadata.json"

# Load Text
print("Reading corpus...")
text = RAW_PATH.read_text(encoding="utf-8")
print(f"Characters : {len(text):,}")

# Tokenization
print("Loading GPT-2 tokenizer...")

tokenizer = Tokenizer()
tokens = tokenizer.encode(text)
print(f"Tokens : {len(tokens):,}")


# Train / Validation Split
split_index = int(len(tokens) * 0.9)
train_tokens = tokens[:split_index]
val_tokens = tokens[split_index:]

# Save Binary Files
np.array(
    train_tokens,
    dtype=np.uint16
).tofile(TRAIN_PATH)

np.array(
    val_tokens,
    dtype=np.uint16
).tofile(VAL_PATH)


# Save Metadata
metadata = {

    "vocab_size": tokenizer.vocab_size,
    "total_tokens": len(tokens),
    "train_tokens": len(train_tokens),
    "validation_tokens": len(val_tokens)
}

with open(META_PATH, "w") as file:
    json.dump(
        metadata,
        file,
        indent=4
    )

print()
print("Processing Complete!")
print(f"Train Tokens      : {len(train_tokens):,}")
print(f"Validation Tokens : {len(val_tokens):,}")
print()
print("Files Created")

print(TRAIN_PATH)
print(VAL_PATH)
print(META_PATH)
