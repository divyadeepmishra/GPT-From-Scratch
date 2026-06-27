import tiktoken

class Tokenizer:

    def __init__(self, encoding="gpt2"):
        self.tokenizer = tiktoken.get_encoding(
            encoding
        )

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def __len__(self):
        return self.vocab_size


tokenizer = Tokenizer()

text = "Hello MiniGPT"

tokens = tokenizer.encode(text)

print(tokens)

print(tokenizer.decode(tokens))

print(tokenizer.vocab_size)
