import pandas as pd
import json
import os
import sys


class Tokenizer:
    def __init__(self, text):
        self.text = text

    # TODO: this is very inefficient since we rebuild the hash map each time; figure out
    # how to update in place
    def _find_top_pair(self, tokens):
        """
        Find top occurring pair of tokens, only return if it occurs more than once, else None.
        """
        freq = {}
        sz = len(tokens)
        for i in range(sz - 1):
            pair = (tokens[i], tokens[i + 1])
            freq[pair] = freq.get(pair, 0) + 1

        top_pairs = sorted(
            list([k, v] for k, v in freq.items()), key=lambda x: x[1], reverse=True
        )
        return top_pairs[0][0] if top_pairs and top_pairs[0][1] > 1 else None

    def _replace(self, tokens, reverse):
        """
        Reverse all unicode code points with their equivalent token idx
        """
        cp = []
        for token in tokens:
            if token not in reverse:
                # this should never happen
                raise AssertionError("Error: token not in reverse mapping.")
            cp.append(reverse[token])
        return cp

    def _init_vocab(self, tokens):
        """
        Initialize vocabulary with all distinct unicode characters that appear
        in the list of unicode code points.

        Returns a tuple consisting of a mapping from token idx -> unicode and a reverse
        mapping from unicode -> token idx
        """
        uniq = set(tokens)
        base = {}
        reverse = {}
        for i, token in enumerate(uniq):
            base[i] = token
            reverse[token] = i
        return (base, reverse)

    def _merge(self, tokens, top_pair, next_idx):
        """
        Replace every occurrence of top_pair with next_idx
        """
        cp = []
        sz = len(tokens)
        i = 0
        while i < sz - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair == top_pair:
                cp.append(next_idx)
                i += 2
            else:
                cp.append(tokens[i])
                i += 1
        if i == sz - 1:
            cp.append(tokens[sz - 1])
        return cp

    def _get_freq(self, tokens):
        freq = {}
        for i in tokens:
            freq[i] = freq.get(i, 0) + 1
        return freq

    def _find_symbol(self, base, vocab, key):
        """
        Returns list of unicode code points representing some token "key"
        """
        if key in base:
            return [base[key]]
        fst, snd = vocab[key]
        return self._find_symbol(base, vocab, fst) + self._find_symbol(base, vocab, snd)

    def print_vocab(self, base, vocab):
        """
        Recursively resolve each entry in vocab and print its associated string
        """
        print("[INFO] printing generated vocab...")
        keys = sorted(vocab.keys())
        for key in keys:
            symbol = self._find_symbol(base, vocab, key)
            displ = "{} => |{}|".format(
                str(key).ljust(3),
                "".join([(chr(c) if c != 10 else "\\n") for c in symbol]),
            )
            print(displ)

    def encode(self, max_len=400):
        tokens = [ord(c) for c in self.text]
        base, reverse = self._init_vocab(tokens)
        tokens = self._replace(tokens, reverse)
        next_idx = max(base.keys()) + 1
        i = 1
        vocab = {}
        initial_token_count = len(tokens)
        while len(vocab) < max_len:
            top_pair = self._find_top_pair(tokens)
            if not top_pair:
                break
            vocab[next_idx] = top_pair
            tokens = self._merge(tokens, top_pair, next_idx)
            next_idx += 1
            print(f"[INFO] iteration: {i}, vocab size: {len(vocab)}")
            i += 1
        final_token_count = len(tokens)
        print(
            f"[INFO] BPE compression ratio: {(1 - initial_token_count / final_token_count)}"
        )
        return (tokens, base, vocab)

    def _decode_token(self, token, base, vocab):
        if token in base:
            return [chr(base[token])]
        lookup = vocab[token]
        return self._decode_token(lookup[0], base, vocab) + self._decode_token(
            lookup[1], base, vocab
        )

    def decode(self, tokens, base, vocab):
        res = []
        for token in tokens:
            res.append(self._decode_token(token, base, vocab))
        return "".join([s for li in res for s in li])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 tokenizer.py <directory path> <vocab size>")
        sys.exit(1)

    dirpath = sys.argv[1]
    vocab_sz = int(sys.argv[2])
    file_contents = []

    for root, _, files in os.walk(dirpath):
        for file in files:
            with open(os.path.join(root, file), "r") as fp:
                data = fp.read()
                file_contents.append(data)
    content = " ".join(file_contents)

    tokenizer = Tokenizer(content)
    tokens, base, vocab = tokenizer.encode(max_len=vocab_sz)
    decoded = tokenizer.decode(tokens, base, vocab)
    assert decoded == content
    print(f"[INFO] final vocab size: {len(vocab)}")
    tokenizer.print_vocab(base, vocab)

    with open("./out/tokens.txt", "w+") as fp:
        fp.write(",".join([str(token) for token in tokens]))

    with open("./out/base.json", "w+") as fp:
        json.dump(base, fp)

    with open("./out/vocab.json", "w+") as fp:
        json.dump(vocab, fp)
