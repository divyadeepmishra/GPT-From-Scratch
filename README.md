# MiniGPT-From-Scratch

> Building a GPT-style language model from first principles using PyTorch.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformer](https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-8A2BE2?style=for-the-badge)

</p>

---

## Why this repository?

Large language models often feel like a black box.

You can use them, fine-tune them, or call them through an API, but understanding *why* they work is a completely different challenge.

This project started with a simple objective:

> **Build a GPT model from scratch and understand every component instead of treating it as magic.**

Rather than relying on existing GPT implementations, I wanted to assemble the model one module at a time, understand what each file contributes, train it on a real dataset, and end up with a model capable of generating coherent text.

The result is a compact, educational implementation of a decoder-only Transformer that focuses on clarity without sacrificing the core ideas behind modern language models.

---

## What this project includes

Instead of being a wrapper around an existing implementation, this repository builds every major component required to train a GPT-style language model.

### Model

- Token Embeddings
- Learnable Positional Embeddings
- Multi-Head Causal Self Attention
- Feed Forward Network (GELU)
- Residual Connections
- Layer Normalization
- Decoder-only Transformer
- Weight Tying
- GPT-style Text Generation

### Training Pipeline

- TinyStories Dataset
- GPT-2 Tokenization (`tiktoken`)
- Binary Dataset Preprocessing
- Custom PyTorch Dataset
- DataLoader
- Training Loop
- Validation Loop
- Checkpoint Saving
- Temperature Sampling
- Top-k Sampling

---

## Project Philosophy

This repository is intentionally **small enough to understand**.

The goal isn't to compete with production frameworks or reproduce GPT-4.

Instead, the focus is on building a clean implementation where every file has a clear purpose, every layer can be inspected, and the complete training pipeline can be followed from raw text all the way to generated output.

If you've ever wondered what happens between typing a prompt and getting a response from a language model, this project aims to make that journey much less mysterious.

---

# How it works

The project follows the same high-level workflow as a modern decoder-only language model, starting from raw text and ending with autoregressive text generation.

```text
                        TinyStories Dataset
                               │
                               ▼
                    scripts/download_dataset.py
                               │
                               ▼
                     data/raw/corpus.txt
                               │
                               ▼
                       processing.py
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
 data/processed/train.bin                 data/processed/val.bin
                │                                   │
                └──────────────┬────────────────────┘
                               ▼
                          dataset.py
                               │
                               ▼
                    PyTorch DataLoader
                               │
                               ▼
                           MiniGPT Model
                               │
       ┌───────────────────────┼────────────────────────┐
       │                       │                        │
       ▼                       ▼                        ▼
 Token Embedding     Positional Encoding      Transformer Blocks
                                                    │
                                                    ▼
                                         Multi-Head Self Attention
                                                    │
                                                    ▼
                                            Feed Forward Network
                                                    │
                                                    ▼
                                             Layer Normalization
                                                    │
                                                    ▼
                                                LM Head
                                                    │
                                                    ▼
                                          Next Token Prediction
                                                    │
                                                    ▼
                                            Generated Text
```

---

# Repository Structure

```
GPT-From-Scratch
│
├── config.py                  # Model & training configuration
├── tokenizer.py               # GPT-2 tokenizer wrapper
├── embeddings.py              # Token embedding layer
├── positional_encoding.py     # Learnable positional embeddings
├── self_attention.py          # Single-head attention implementation
├── multi_head_attention.py    # Multi-head causal self-attention
├── feed_forward.py            # Transformer MLP block
├── layer_norm.py              # Layer normalization wrapper
├── transformer_block.py       # Complete decoder block
├── model.py                   # MiniGPT architecture
│
├── dataset.py                 # Binary token dataset
├── processing.py              # Dataset preprocessing pipeline
├── trainer.py                 # Model training
├── generate.py                # Text generation
│
├── scripts/
│   └── download_dataset.py    # Downloads and prepares TinyStories
│
├── data/
│   ├── raw/
│   └── processed/
│
├── checkpoints/
│
└── outputs/
```

---

# Inside the model

The model is intentionally modular.

Rather than writing everything inside a single file, each major Transformer component lives in its own module.

This makes it easier to inspect individual layers, experiment with architectural changes, and understand how information flows through the network.

```
Input Tokens
      │
      ▼
Token Embedding
      │
      ▼
Positional Embedding
      │
      ▼
┌────────────────────────────┐
│   Transformer Block × N    │
│                            │
│ LayerNorm                  │
│      ↓                     │
│ Multi-Head Attention        │
│      ↓                     │
│ Residual Connection         │
│      ↓                     │
│ LayerNorm                  │
│      ↓                     │
│ Feed Forward Network        │
│      ↓                     │
│ Residual Connection         │
└────────────────────────────┘
      │
      ▼
Final LayerNorm
      │
      ▼
Language Modeling Head
      │
      ▼
Vocabulary Logits
```

---

# Design Decisions

A few implementation choices were made deliberately.

### GPT-2 Tokenizer

Instead of implementing Byte Pair Encoding from scratch, this project uses OpenAI's `tiktoken` tokenizer. The objective here is to study Transformer internals rather than recreate an industrial tokenizer.

---

### Learnable Positional Embeddings

The model uses learnable positional embeddings similar to early GPT models. This keeps the implementation approachable while matching the architecture the project set out to reproduce.

---

### Decoder-only Architecture

The model predicts the next token using masked self-attention.

Every token can only attend to tokens that came before it, making autoregressive generation possible.

---

### Weight Tying

The token embedding matrix and the output projection layer share the same weights.

Besides reducing the parameter count, this follows the design used in many modern language models and slightly improves training efficiency.

---

### Readability over cleverness

Some implementations optimize aggressively.

This one intentionally doesn't.

Whenever there was a trade-off between writing clever code and writing understandable code, readability won.

---

# Dataset

Training a language model starts long before the first forward pass.

Instead of feeding raw text directly into the model, the dataset is first converted into a compact binary representation that can be streamed efficiently during training.

For this project, the **TinyStories** dataset is used as the training corpus. It provides thousands of short English stories that are ideal for experimenting with autoregressive language models while keeping training time manageable.

The complete preprocessing pipeline looks like this:

```text
TinyStories
      │
      ▼
download_dataset.py
      │
      ▼
corpus.txt
      │
      ▼
GPT-2 Tokenization
      │
      ▼
train.bin
val.bin
metadata.json
```

After preprocessing, the model never reads raw text again.

Instead, it trains directly on binary token files, which makes loading significantly faster and reduces unnecessary preprocessing during every epoch.

---

# Training Configuration

The final model was trained using the following configuration.

| Parameter | Value |
|-----------|------:|
| Parameters | **≈ 44.7M** |
| Vocabulary Size | 50,257 |
| Context Length | 128 Tokens |
| Layers | 6 |
| Attention Heads | 8 |
| Embedding Dimension | 512 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Epochs | 5 |

---

# Training Results

After five epochs, the model converged to the following metrics.

| Metric | Value |
|---------|------:|
| Train Loss | **2.5111** |
| Validation Loss | **2.5752** |

The relatively small gap between training and validation loss suggests that the model learned meaningful language patterns without obvious signs of heavy overfitting during the initial training run.

---

# Running the Project

## 1. Clone the repository

```bash
git clone https://github.com/divyadeepmishra/GPT-From-Scratch.git

cd GPT-From-Scratch
```

---

## 2. Install dependencies

```bash
pip install torch tiktoken datasets tqdm numpy
```

---

## 3. Download the dataset

```bash
python scripts/download_dataset.py
```

This creates

```text
data/raw/corpus.txt
```

---

## 4. Preprocess the dataset

```bash
python processing.py
```

Generated files

```text
train.bin
val.bin
metadata.json
```

---

## 5. Train the model

```bash
python trainer.py
```

During training the script

- Loads the binary dataset
- Creates DataLoaders
- Performs forward and backward passes
- Computes validation loss
- Saves the best checkpoint automatically

The best model is stored inside

```text
checkpoints/minigpt.pt
```

---

## 6. Generate text

```bash
python generate.py
```

Example

```text
Prompt:

Once upon a time
```

Generated Output

```text
Once upon a time, there was a black girl named Jill.
She was three years old, but she was curious.
She was always in her garden.

One day...
```

Every generated sample is also written to the **outputs/** directory for later inspection.

---

# Example Prompts

Some prompts that work well with the TinyStories dataset.

```text
Once upon a time

A little rabbit

There was a boy named

The little dragon

One sunny morning

The happy elephant
```

Since the model was trained entirely on TinyStories, prompts written in a similar storytelling style generally produce the most coherent generations.

---

# Engineering Notes

One thing this project reinforced is that large language models are built from a collection of surprisingly simple ideas.

None of the individual components are overwhelmingly complicated on their own. The challenge comes from understanding how they interact.

Building the model from scratch made concepts like attention, residual connections, positional embeddings, and autoregressive generation much easier to reason about than simply reading papers or using high-level libraries.

Several implementation decisions were also intentionally kept simple.

For example, the project uses OpenAI's GPT-2 tokenizer instead of implementing Byte Pair Encoding from scratch. While building a tokenizer is an interesting problem on its own, the objective here was to spend more time understanding Transformer internals rather than text compression algorithms.

Similarly, the implementation favors readability over aggressive optimization. There are many ways to make a Transformer faster, but a clear implementation is often a better starting point for experimentation.

The current codebase represents a solid baseline that can be extended without becoming difficult to understand.

---

# Current Limitations

Like any first implementation, this project makes a few deliberate trade-offs.

- It uses learnable positional embeddings instead of Rotary Positional Embeddings (RoPE).
- Standard scaled dot-product attention is used instead of Flash Attention.
- Training is performed in full precision without mixed-precision optimization.
- The model is trained on TinyStories, so its writing style naturally reflects the dataset it has seen.
- Long-term context is limited by the configured context window.

None of these choices are accidental. They keep the implementation approachable while preserving the core mechanics of a GPT-style decoder.

---

# Future Improvements

Rather than continuously modifying the existing implementation, future work will focus on building separate versions that explore more modern techniques.

Some ideas include:

- Rotary Position Embeddings (RoPE)
- Flash Attention
- KV Cache for faster inference
- Mixed Precision Training
- Learning Rate Scheduler
- Gradient Accumulation
- LoRA Fine-Tuning
- Instruction Tuning
- Hugging Face dataset integration
- Interactive web interface using Gradio or Streamlit

Keeping these improvements separate allows Version 1 to remain a clean educational implementation while future versions focus on performance and experimentation.

---

# Contributing

Suggestions, improvements, and discussions are always welcome.

If you notice a bug, have an idea for a cleaner implementation, or want to experiment with a new architectural component, feel free to open an issue or submit a pull request.

---

## Final Thoughts

Building a language model from scratch is less about recreating state-of-the-art systems and more about understanding the ideas that make them possible.

This repository is my attempt to bridge the gap between reading about Transformers and actually implementing one.

There is still plenty of room for improvement, but that's also what makes projects like this enjoyable. Every experiment raises new questions, and every iteration reveals another layer of how modern language models work.

If this repository helps someone take their first step into understanding GPT-style models, then it has already accomplished more than I originally expected.

⭐ If you found this project useful, consider giving it a star. It helps others discover the repository and encourages future improvements.

