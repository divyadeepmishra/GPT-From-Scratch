class MiniBPE:

    def __init__(self):
        self.vocab = {}


    def get_pairs(self, words):
        pairs = []
        for i in range(len(word)-1):
            pairs.append(
                (
                    word[i],
                    word[i+1]
                )
            )
        return pairs

    def count_pairs(self, words):
        pair_counts = {}

        for word in words:
            pairs = self.get_pairs(word)
            for pair in pairs:
                if pair not in pair_counts:
                    pair_counts[pair] = 1
                else:
                    pair_counts += 1
                    
    return pair_counts
