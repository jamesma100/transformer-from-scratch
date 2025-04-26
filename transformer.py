import torch
import torch.nn as nn
import math
import csv
from tokenizer import Tokenizer
from collections.abc import Callable

from typing import List, Dict


class Embedding(nn.Module):
    # NOTE: This is a bit memory inefficient if you pass in the full vocab, since
    # many tokens in the full vocab will not be present in the input string. You can
    # alleviate this by compressing the vocab s.t. it contains only the tokens that
    # are in the input. Doing so, however, means you need to rescale your input s.t.
    # each token lie in range [0, length of compressed vocab - 1] as opposed to
    # [0, length of full vocab - 1] otherwise it will not be a valid embedding
    def __init__(self, vocab_sz: int, d_model: int):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_sz, d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.embedding(X) * math.sqrt(self.d_model)

class PosEnc(nn.Module):
    def __init__(self, n: int, d_model: int, base: int = 10000, p_drop: float = 0.1):
        super(PosEnc, self).__init__()
        # TODO: make this not dumb
        self.PE = torch.zeros(n, d_model)
        for pos in range(n):
            for i in range(d_model // 2):
                denom = math.pow(10000, 2*i/d_model)
                self.PE[pos][2*i] = math.sin(pos / denom)
                self.PE[pos][2*i+1] = math.cos(pos / denom)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X + self.dropout(self.PE)


class LayerNorm(nn.Module):
    def __init__(self, n: int, d_model: int, eps: float = 1e-5):
        super(LayerNorm, self).__init__()
        self.n = n
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(n, d_model))
        self.beta = nn.Parameter(torch.zeros(n, d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean = X.mean(1, keepdim=True)
        var = X.var(1, keepdim=True, unbiased=False)
        return (X - mean) / torch.sqrt(var + self.eps) * self.alpha + self.beta


class ResidualConnection(nn.Module):
    def __init__(self, p_drop: float = 0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X: torch.Tensor, sublayer: Callable[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return X + self.dropout(sublayer(X))

class AddAndNorm(nn.Module):
    def __init__(self, n: int, d_model: int, eps: float =1e-5, p_drop: float = 0.1):
        super(AddAndNorm, self).__init__()
        self.add = ResidualConnection()
        self.norm = LayerNorm(n, d_model, eps)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X: torch.Tensor, sublayer: Callable[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.norm(self.add(X, sublayer))


class FFN(nn.Module):
    """
    y = xAT + b
    Input: X(nx512), W1(512x2048), W2(2048x512), b1(1x2048), b2(1x512)
    Output: max{0, (xW1 + b1)}W2 + b2 (nx512)
    """
    def __init__(self, n: int, d_model: int, dff: int):
        super(FFN, self).__init__()
        self.W1 = nn.Linear(d_model, dff)
        self.W2 = nn.Linear(dff, d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.W2(self.W1(X).relu())


class MultiheadedAttention(nn.Module):
    def __init__(self, n: int, h: int, d_model: int):
        super(MultiheadedAttention, self).__init__()
        self.n = n
        self.h = h
        self.d_model = d_model
        self.d_q = d_model // h
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.queries = [nn.Linear(d_model, self.d_q) for i in range(h)]
        self.keys = [nn.Linear(d_model, self.d_k) for i in range(h)]
        self.values = [nn.Linear(d_model, self.d_v) for i in range(h)]

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        QKT = Q.matmul(K.transpose(0, 1))
        return (QKT / math.sqrt(self.d_k)).softmax(dim=1).matmul(V)

    # NOTE: X_Q = X_K = X_V if self-attention
    def forward(self, X_Q: torch.Tensor, X_K: torch.Tensor, X_V: torch.Tensor) -> torch.Tensor:
        Qs = [self.queries[i](X_Q) for i in range(self.h)]
        Ks = [self.keys[i](X_K) for i in range(self.h)]
        Vs = [self.values[i](X_V) for i in range(self.h)]
        heads = [self.attention(Q, K, V) for Q, K, V in zip(Qs, Ks, Vs)]
        return torch.cat(heads, dim=1)


class EncoderLayer(nn.Module):
    def __init__(self, n: int, h: int, d_model: int):
        super(EncoderLayer, self).__init__()
        self.n = n
        self.h = h
        self.d_model = d_model
        self.layer1 = MultiheadedAttention(n, h, d_model)
        self.add_and_norm1 = AddAndNorm(n, d_model)
        self.layer2 = FFN(n, d_model, d_model * 4)
        self.add_and_norm2 = AddAndNorm(n, d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        attention_out = self.add_and_norm1(X, lambda x: self.layer1(x, x, x))
        ffn_out = self.add_and_norm2(attention_out, lambda x: self.layer2(x))
        return ffn_out


class DecoderLayer(nn.Module):
    def __init__(self, n: int, h: int, d_model: int, Z_e: torch.Tensor):
        super(DecoderLayer, self).__init__()
        self.n = n
        self.h = h
        self.d_model = d_model
        self.Z_e = Z_e

        self.layer1 = MultiheadedAttention(n, h, d_model)
        self.add_and_norm1 = AddAndNorm(n, d_model)
        self.layer2 = MultiheadedAttention(n, h, d_model)
        self.add_and_norm2 = AddAndNorm(n, d_model)
        self.layer3 = FFN(n, d_model, d_model * 4)
        self.add_and_norm3 = AddAndNorm(n, d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self_attention_out = self.add_and_norm1(X, lambda x: self.layer1(x, x, x))
        cross_attention_out = self.add_and_norm2(self_attention_out, lambda x: self.layer2(x, self.Z_e, self.Z_e))
        ffn_out = self.add_and_norm3(cross_attention_out, lambda x: self.layer3(x))
        return ffn_out


class Encoder(nn.Module):
    def __init__(self, n: int, h: int, d_model: int, num_layers: int):
        super(Encoder, self).__init__()
        self.pe = PosEnc(n, d_model)
        self.encoder_layers = [EncoderLayer(n, h, d_model) for i in range(num_layers)]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z_e = self.pe(X)
        for encoder_layer in self.encoder_layers:
            Z_e = encoder_layer(Z_e)
        return Z_e


class Decoder(nn.Module):
    def __init__(self, n: int, h: int, d_model: int, num_layers: int, Z_e: torch.Tensor):
        super(Decoder, self).__init__()
        self.Z_e = Z_e
        self.pe = PosEnc(n, d_model)
        self.decoder_layers = [DecoderLayer(n, h, d_model, Z_e) for i in range(num_layers)]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z_d = self.pe(X)
        for decoder_layer in self.decoder_layers:
            Z_d = decoder_layer(Z_d)
        return Z_d

def tokenize_input(t_e: List[str], d_model: int) -> torch.Tensor:
    src_tokenizer = Tokenizer(t_e)
    src_tokens, src_vocab = src_tokenizer.encode()
    src_embedding_fn = Embedding(len(src_vocab), d_model)
    src_embedding = src_embedding_fn(torch.tensor(src_tokens[0]))
    return src_embedding


def tokenize_output(t_d: List[str], d_model: int) -> torch.Tensor:
    # return torch.randn(100, 512)
    tgt_tokenizer = Tokenizer(t_d)
    tgt_tokens, tgt_vocab = tgt_tokenizer.encode()
    tgt_embedding_fn = Embedding(len(tgt_vocab), d_model)
    tgt_embedding = tgt_embedding_fn(torch.tensor(tgt_tokens[0]))
    print("tgt: ", tgt_embedding)
    return tgt_embedding

# TODO(implement)
def decoder_unembed(Z_d: torch.Tensor) -> torch.Tensor:
    return Z_d

def simulate(t_e: List[str], t_d: List[str], d_model: int, h: int, num_layers: int) -> None:
    # <Encoder start>
    Z_e = tokenize_input(t_e, d_model)
    n_src = Z_e.size()[0]
    encoder = Encoder(n_src, h, d_model, 6)
    Z_e = encoder(Z_e)

    Z_d = tokenize_output(t_d, d_model)
    n_tgt = Z_d.size()[0]
    decoder = Decoder(n_tgt, h, d_model, 6, Z_e)
    Z_d = decoder(Z_d)

    out = decoder_unembed(Z_d)
    return out


if __name__ == "__main__":
    de, en = [], []
    with open("./data/sample.csv", newline='') as f:
        reader = csv.reader(f)
        for i, l in enumerate(reader):
            if i == 0: # skip title row
                continue
            de.append(l[0])
            en.append(l[1])

    d_model = 512

    src_tokenizer = Tokenizer(de)
    src_tokens, src_vocab = src_tokenizer.encode()
    src_embedding_fn = Embedding(len(src_vocab), d_model)
    src_embeddings = [src_embedding_fn(torch.tensor(sentence)) for sentence in src_tokens]

    tgt_tokenizer = Tokenizer(en)
    tgt_tokens, tgt_vocab = tgt_tokenizer.encode()
    tgt_embedding_fn = Embedding(len(tgt_vocab), d_model)
    tgt_embeddings = [src_embedding_fn(torch.tensor(sentence)) for sentence in tgt_tokens]

    out = simulate([de[1]], [en[1]], d_model, 8, 6)
    print(out.size())

