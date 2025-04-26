import csv

class Tokenizer:
    def __init__(self, text):
        self.text = text

    def _find_top_pair(self, tokens):
        freq = {}
        for l in tokens:
            sz = len(l)
            for i in range(sz - 1):
                pair = (l[i], l[i+1])
                freq[pair] = freq.get(pair, 0) + 1

        top_pairs = sorted(list([k, v] for k, v in freq.items()), key=lambda x:x[1], reverse=True)
        return top_pairs[0][0] if top_pairs and top_pairs[0][1] > 1 else None
    
    def _replace(self, tokens, reverse):
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


    def _merge(self, tokens, top_pair, next_idx):
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
    
    def _find_symbol(self, vocab, val):
        if len(val) == 1:
            return val
        fst, snd = val
        return self._find_symbol(vocab, vocab[fst]) + self._find_symbol(vocab, vocab[snd])
    
    def print_vocab(self, vocab):
        keys = sorted(vocab.keys())
        for key in keys:
            symbol = self._find_symbol(vocab, vocab[key])
            displ = "{} => |{}|".format(
                str(key).ljust(3),
                ''.join([(chr(c) if c != 10 else "\\n") for c in symbol])
            )
            print(displ)
    
    def _init_vocab(self, tokens):
        vocab = {}
        inverted = {}
        next_idx = 0
        for l in tokens:
            for c in l:
                if c in inverted:
                    continue
                vocab[next_idx] = (c,)
                inverted[c] = next_idx
                next_idx += 1
        return (vocab, inverted)
    
    def encode(self, max_len=1000):
        tokens = [[ord(c) for c in l] for l in self.text]
        vocab, inverted = self._init_vocab(tokens)
        tokens = self._replace(tokens, inverted)
        next_idx = max(vocab.keys()) + 1
        while len(vocab) < max_len:
            top_pair = self._find_top_pair(tokens)
            if not top_pair:
                break
            vocab[next_idx] = top_pair
            tokens = self._merge(tokens, top_pair, next_idx)
            next_idx += 1
        return (tokens, vocab)
    
    def _decode_token(self, token, vocab):
        lookup = vocab[token]
        if len(lookup) == 1:
            return [chr(lookup[0])]
        return self._decode_token(lookup[0], vocab) + self._decode_token(lookup[1], vocab)

    def decode(self, tokens, vocab):
        res = []
        for l in tokens:
            tmp = []
            for token in l:
                tmp.append("".join(self._decode_token(token, vocab)))
            res.append("".join(tmp))
        return res


if __name__ == "__main__":
    text = """
Je m’appelle Jessica
Je m’appelle Jessica. Je suis une fille, je suis française et j’ai treize ans. Je vais à l’école à Nice, mais j’habite à Cagnes-Sur-Mer. J’ai deux frères. Le premier s’appelle Thomas, il a quatorze ans. Le second s’appelle Yann et il a neuf ans. Mon papa est italien et il est fleuriste. Ma mère est allemande et est avocate. Mes frères et moi parlons français, italien et allemand à la maison. Nous avons une grande maison avec un chien, un poisson et deux chats.

Aujourd’hui, on est samedi, nous rendons visite à notre grand-mère. Elle a 84 ans et elle habite à Antibes. J’adore ma grand-mère, elle est très gentille. Elle fait des bons gâteaux.

Lundi, je retourne à l’école. Je suis contente, je vais voir Amélie. C’est ma meilleure amie. J’aime beaucoup l’école. Mes matières préférées sont le français et le sport. J’aime beaucoup lire et je nage très bien."""
    de, en = [], []
    with open("./data/sample.csv") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            de.append(row[0])
            en.append(row[1])
    tokenizer = Tokenizer(en)
    tokens, vocab = tokenizer.encode()
    decoded = tokenizer.decode(tokens, vocab)
    assert decoded == en
