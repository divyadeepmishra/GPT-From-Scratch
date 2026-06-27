class SimpleTokenizer:

    def __init__(self, text):
        words = text.split()
        vocab = sorted(set(words))

        self.word_to_id = {
            word: idx
            for idx, word in enumerate(vocab)
        }
        self.id_to_word = {
            idx: word
            for word, idx in self.word_to_id.items()
        }

    def encode(self, text):
        return[
            self.word_to_id[word]
            for word in text.split()
        ]

    def decode(self, ids):
        return " ".join(
            self.id_to_word[idx]
            for idx in ids
        )

text = """
hello buddy
welcome to gpt
hello world
"""

tokenizer = SimpleTokenizer(text)

tokens = tokenizer.encode("hello world")

print(tokens)

print(tokenizer.decode(tokens))
