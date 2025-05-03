import pandas as pd
import json

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
        for l in tokens:
            sz = len(l)
            for i in range(sz - 1):
                pair = (l[i], l[i+1])
                freq[pair] = freq.get(pair, 0) + 1

        top_pairs = sorted(list([k, v] for k, v in freq.items()), key=lambda x:x[1], reverse=True)
        return top_pairs[0][0] if top_pairs and top_pairs[0][1] > 1 else None
    
    def _replace(self, tokens, reverse):
        """
        Reverse all unicode code points with their equivalent token idx
        """
        cp = []
        for l in tokens:
            tmp = []
            for token in l:
                if token not in reverse:
                    # this should never happen
                    raise AssertionError("Error: token not in reverse mapping.")
                tmp.append(reverse[token])
            cp.append(tmp)
        return cp
    
    def _init_vocab(self, tokens):
        """
        Initialize vocabulary with all distinct unicode characters that appear
        in the list of unicode code points.

        Returns a tuple consisting of a mapping from token idx -> unicode and a reverse
        mapping from unicode -> token idx
        """
        base = {}
        reverse = {}
        next_idx = 1  # save 0 for padding
        for l in tokens:
            for c in l:
                if c in reverse:
                    continue
                base[next_idx] = c
                reverse[c] = next_idx
                next_idx += 1
        return (base, reverse)

    def _merge(self, tokens, top_pair, next_idx):
        """
        Replace every occurrence of top_pair with next_idx
        """
        cp = []
        for l in tokens:
            sz = len(l)
            i = 0
            tmp = []
            while i < sz - 1:
                pair = (l[i], l[i+1])
                if pair == top_pair:
                    tmp.append(next_idx)
                    i += 2
                else:
                    tmp.append(l[i])
                    i += 1
            if i == sz - 1:
                tmp.append(l[sz - 1])
            cp.append(tmp)
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
        keys = sorted(vocab.keys())
        for key in keys:
            symbol = self._find_symbol(base, vocab, key)
            displ = "{} => |{}|".format(
                str(key).ljust(3),
                ''.join([(chr(c) if c != 10 else "\\n") for c in symbol])
            )
            print(displ)
    
    
    def encode(self, max_len=3000):
        tokens = [[ord(c) for c in l] for l in self.text]
        base, reverse = self._init_vocab(tokens)
        tokens = self._replace(tokens, reverse)
        next_idx = max(base.keys()) + 1
        i = 1
        vocab = {}
        initial_token_count = sum(len(sentence) for sentence in tokens)
        while len(vocab) < max_len:
            top_pair = self._find_top_pair(tokens)
            if not top_pair:
                break
            vocab[next_idx] = top_pair
            tokens = self._merge(tokens, top_pair, next_idx)
            next_idx += 1
            i += 1
        final_token_count = sum(len(sentence) for sentence in tokens)
        print("[INFO] BPE compression ratio: {}".format(final_token_count / initial_token_count))
        return (tokens, base, vocab)
    
    def _decode_token(self, token, base, vocab):
        if token in base:
            return [chr(base[token])]
        lookup = vocab[token]
        return self._decode_token(lookup[0], base, vocab) + self._decode_token(lookup[1], base, vocab)

    def decode(self, tokens, base, vocab):
        res = []
        for l in tokens:
            tmp = []
            for token in l:
                tmp.append("".join(self._decode_token(token, base, vocab)))
            res.append("".join(tmp))
        return res


if __name__ == "__main__":
    nrows = 200
    en = []
    fr = []
    data = pd.read_csv('./data/wmt14_translate_fr-en_train_sample.csv', lineterminator='\n', nrows=nrows)
    for i, row in data.iterrows():
        en.append(row.iloc[1])
        fr.append(row.iloc[0])

    tokenizer_en = Tokenizer(en)
    tokens_en, base_en, vocab_en = tokenizer_en.encode()
    decoded_en = tokenizer_en.decode(tokens_en, base_en, vocab_en)
    assert decoded_en == en

    with open("./out/tokens_en.txt", "w+") as fp:
        for tokens in tokens_en:
            fp.write("{}\n".format(",".join([str(token) for token in tokens])))

    with open("./out/base_en.json", "w+") as fp:
        json.dump(base_en, fp)

    with open("./out/vocab_en.json", "w+") as fp:
        json.dump(vocab_en, fp)

    tokenizer_fr = Tokenizer(fr)
    tokens_fr, base_fr, vocab_fr = tokenizer_fr.encode()
    decoded_fr = tokenizer_fr.decode(tokens_fr, base_fr, vocab_fr)
    assert decoded_fr == fr

    with open("./out/tokens_fr.txt", "w+") as fp:
        for tokens in tokens_fr:
            fp.write("{}\n".format(",".join([str(token) for token in tokens])))

    with open("./out/base_fr.json", "w+") as fp:
        json.dump(base_fr, fp)

    with open("./out/vocab_fr.json", "w+") as fp:
        json.dump(vocab_fr, fp)
