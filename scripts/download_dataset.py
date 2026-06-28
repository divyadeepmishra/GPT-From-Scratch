from pathlib import Path
from datasets import load_dataset

MAX_SIZE_MB = 8

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = RAW_DIR / "corpus.txt"

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True
)

print("Creating corpus.txt...\n")

current_size = 0
story_count = 0

with open(CORPUS_PATH, "w", encoding="utf-8") as file:

    for story in dataset:

        text = story["text"].strip()

        file.write(text)
        file.write("\n\n")

        current_size = file.tell()

        story_count += 1

        if story_count % 100 == 0:
            print(f"{story_count} stories collected...")

        if current_size >= MAX_SIZE_MB * 1024 * 1024:
            break

print("\nDone!")
print(f"Stories saved : {story_count}")
print(f"Corpus size   : {current_size / (1024*1024):.2f} MB")
print(f"Saved to      : {CORPUS_PATH}")
