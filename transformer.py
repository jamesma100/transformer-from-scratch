from tokenizer import Tokenizer
from typing import List, Dict

import csv
import json
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Size defs

N         Number of tokens in input sentence
D_MODEL   Size of embedding of each token
H         Number of attention heads
D_HEAD    Size of attention head, equal to D_MODEL / H
BATCH_SZ  Size of batch
"""


class Embedding(nn.Module):
    # NOTE: This is a bit memory inefficient if you pass in the full vocab, since
    # many tokens in the full vocab will not be present in the input string. You can
    # alleviate this by compressing the vocab s.t. it contains only the tokens that
    # are in the input. Doing so, however, means you need to rescale your input s.t.
    # each token lie in range [0, length of compressed vocab - 1] as opposed to
    # [0, length of full vocab - 1] otherwise it will not be a valid embedding
    def __init__(self, vocab_sz, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_sz, d_model)

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        return self.embedding(X) * math.sqrt(self.d_model)


class PosEnc(nn.Module):
    def __init__(self, n, d_model, base=10000, p_drop=0.1):
        super(PosEnc, self).__init__()
        self.n = n
        self.PE = torch.zeros(n, d_model)
        for pos in range(n):
            for i in range(d_model // 2):
                denom = math.pow(10000, 2 * i / d_model)
                self.PE[pos][2 * i] = math.sin(pos / denom)
                self.PE[pos][2 * i + 1] = math.cos(pos / denom)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        return X + self.dropout(self.PE)


class LayerNorm(nn.Module):
    def __init__(self, n, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(n, d_model))
        self.beta = nn.Parameter(torch.zeros(n, d_model))

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        mean = X.mean(1, keepdim=True)
        var = X.var(1, keepdim=True, unbiased=False)
        return (X - mean) / torch.sqrt(var + self.eps) * self.alpha + self.beta


class ResidualConnection(nn.Module):
    def __init__(self, p_drop=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X, sublayer):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)
        sublayer: function mapping tensor -> tensor, both of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        return X + self.dropout(sublayer(X))


class AddAndNorm(nn.Module):
    def __init__(self, n, d_model, eps=1e-5, p_drop=0.1):
        super(AddAndNorm, self).__init__()
        self.add = ResidualConnection()
        self.norm = LayerNorm(n, d_model, eps)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X, sublayer):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)
        sublayer: function mapping tensor -> tensor, both of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        return self.norm(self.add(X, sublayer))


class FFN(nn.Module):
    def __init__(self, n, d_model, dff):
        super(FFN, self).__init__()
        self.W1 = nn.Linear(d_model, dff)
        self.W2 = nn.Linear(dff, d_model)

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        return self.W2(self.W1(X).relu())


# TODO add masking to encoder to skip padding
# TODO add masking to decoder to skip later tokens
class MultiheadedAttention(nn.Module):
    def __init__(self, n, h, d_model, key_padding_mask=None, att_mask=False):
        """
        Args:
        key_padding_mask: tensor of size (BATCH_SZ, N, N)
        att_mask: whether or not illegal connections should be masked
            This should be set to True in the decoder self-attention block
        """
        super(MultiheadedAttention, self).__init__()
        self.n = n
        self.h = h
        self.d_q = d_model // h
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.queries = [nn.Linear(d_model, self.d_q) for i in range(h)]
        self.keys = [nn.Linear(d_model, self.d_k) for i in range(h)]
        self.values = [nn.Linear(d_model, self.d_v) for i in range(h)]
        self.key_padding_mask = key_padding_mask
        self.att_mask = att_mask

    def attention(self, Q, K, V):
        """
        Args:
        Q: tensor of size (BATCH_SZ, N, D_HEAD)
        K: tensor of size (BATCH_SZ, N, D_HEAD)
        V: tensor of size (BATCH_SZ, N, D_HEAD)

        Output: tensor of size (BATCH_SZ, N, D_HEAD)
        """
        QKT = Q.matmul(K.transpose(1, 2))
        if self.key_padding_mask is not None:
            QKT = QKT.masked_fill(self.key_padding_mask, -torch.inf)

        if self.att_mask:
            indices = torch.triu_indices(self.n, self.n, offset=1)
            QKT[:, indices[0], indices[1]] = float("-inf")
        return (QKT / math.sqrt(self.d_k)).softmax(dim=2).matmul(V)

    # NOTE: X_Q = X_K = X_V if self-attention
    def forward(self, X_Q, X_K, X_V):
        """
        Args:
        X_Q: tensor of size (BATCH_SZ, N, D_MODEL)
        X_K: tensor of size (BATCH_SZ, N, D_MODEL)
        X_V: tensor of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        Qs = [self.queries[i](X_Q) for i in range(self.h)]
        Ks = [self.keys[i](X_K) for i in range(self.h)]
        Vs = [self.values[i](X_V) for i in range(self.h)]
        heads = [self.attention(Q, K, V) for Q, K, V in zip(Qs, Ks, Vs)]
        return torch.cat(heads, dim=2)


class EncoderLayer(nn.Module):
    def __init__(self, n, h, d_model, key_padding_mask):
        super(EncoderLayer, self).__init__()
        self.layer1 = MultiheadedAttention(
            n, h, d_model, key_padding_mask=key_padding_mask
        )
        self.add_and_norm1 = AddAndNorm(n, d_model)
        self.layer2 = FFN(n, d_model, d_model * 4)
        self.add_and_norm2 = AddAndNorm(n, d_model)

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N_SRC, D_MODEL)

        Output: tensor of size (BATCH_SZ, N_SRC, D_MODEL)
        """
        attention_out = self.add_and_norm1(X, lambda x: self.layer1(x, x, x))
        ffn_out = self.add_and_norm2(attention_out, lambda x: self.layer2(x))
        return ffn_out


class DecoderLayer(nn.Module):
    def __init__(self, n, h, d_model, Z_e, key_padding_mask):
        """
        Args:
        Z_e: tensor of size (BATCH_SZ, N_SRC, D_MODEL)
            Represents the output of the encoder
        """
        super(DecoderLayer, self).__init__()
        self.Z_e = Z_e

        self.layer1 = MultiheadedAttention(
            n, h, d_model, key_padding_mask=key_padding_mask, att_mask=True
        )
        self.add_and_norm1 = AddAndNorm(n, d_model)
        self.layer2 = MultiheadedAttention(n, h, d_model)  # cross attention
        self.add_and_norm2 = AddAndNorm(n, d_model)
        self.layer3 = FFN(n, d_model, d_model * 4)
        self.add_and_norm3 = AddAndNorm(n, d_model)

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N_TGT, D_MODEL)

        Output: tensor of size (BATCH_SZ, N_TGT, D_MODEL)
        """
        self_attention_out = self.add_and_norm1(X, lambda x: self.layer1(x, x, x))
        cross_attention_out = self.add_and_norm2(
            self_attention_out, lambda x: self.layer2(x, self.Z_e, self.Z_e)
        )
        ffn_out = self.add_and_norm3(cross_attention_out, lambda x: self.layer3(x))
        return ffn_out


class Encoder(nn.Module):
    def __init__(self, n, h, d_model, num_layers, key_padding_mask):
        super(Encoder, self).__init__()
        self.pe = PosEnc(n, d_model)
        self.encoder_layers = [
            EncoderLayer(n, h, d_model, key_padding_mask) for i in range(num_layers)
        ]

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N_SRC, D_MODEL)

        Output: tensor of size (BATCH_SZ, N_SRC, D_MODEL)
        """
        Z_e = self.pe(X)
        for encoder_layer in self.encoder_layers:
            Z_e = encoder_layer(Z_e)
        return Z_e


class Decoder(nn.Module):
    def __init__(self, n, h, d_model, num_layers, Z_e, key_padding_mask):
        """
        Args:
        Z_e: tensor of size (BATCH_SZ, N_SRC, D_MODEL)
            Represents the output of the encoder
        """
        super(Decoder, self).__init__()
        self.Z_e = Z_e
        self.pe = PosEnc(n, d_model)
        self.decoder_layers = [
            DecoderLayer(n, h, d_model, Z_e, key_padding_mask)
            for i in range(num_layers)
        ]

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N_TGT, D_MODEL)

        Output: tensor of size (BATCH_SZ, N_TGT, D_MODEL)
        """
        Z_d = self.pe(X)
        for decoder_layer in self.decoder_layers:
            Z_d = decoder_layer(Z_d)
        return Z_d


def tokenize_input(t_e: List[str], d_model):
    """
    Args:
    t_e: list of sentences, with length BATCH_SZ

    Output: tensor of size (BATCH_SZ, N_SRC, D_MODEL)
        Each sentence is converted into a list of tokens, then padded to the
        max length of the token lists (N_SRC) , then converted to an embedding
        of size D_MODEL
    """
    src_tokenizer = Tokenizer(t_e)
    src_tokens, src_base, src_vocab = src_tokenizer.encode()
    src_tokens = [torch.tensor(sentence) for sentence in src_tokens]
    max_len = max([sentence.size(0) for sentence in src_tokens])

    # tensor of size (BATCH_SZ, max_len) where element at (i, j) == 1 if padded else 0
    key_padding_mask = torch.zeros(len(src_tokens), max_len, dtype=torch.bool)

    # list of size BATCH_SZ of tensors of size (max_len)
    padded = [torch.zeros((1, max_len), dtype=int) for i in range(len(src_tokens))]
    for i in range(len(padded)):
        padded[i][:, : src_tokens[i].size(0)] = src_tokens[i]
        key_padding_mask[i][src_tokens[i].size(0) - 1 :] = 1

    key_padding_mask = key_padding_mask.unsqueeze(1)
    key_padding_mask = key_padding_mask.expand(-1, max_len, -1)

    cat = torch.concat(padded)

    src_embedding_fn = Embedding(len(src_vocab) + len(src_base) + 1, d_model)
    src_embeddings = src_embedding_fn(cat)
    return (src_embeddings, key_padding_mask)


def tokenize_output(t_d: List[str], d_model):
    """
    Args:
    t_d: list of sentences, with length BATCH_SZ

    Output: tensor of size (BATCH_SZ, N_TGT, D_MODEL)
        Each sentence is converted into a list of tokens, then padded to the
        max length of the token lists (N_SRC) , then converted to an embedding
        of size D_MODEL
    """
    tgt_tokenizer = Tokenizer(t_d)
    tgt_tokens, tgt_base, tgt_vocab = tgt_tokenizer.encode()
    tgt_tokens = [torch.tensor(sentence) for sentence in tgt_tokens]
    max_len = max([sentence.size(0) for sentence in tgt_tokens])

    # tensor of size (BATCH_SZ, max_len) where element at (i, j) == 1 if padded else 0
    key_padding_mask = torch.zeros(len(tgt_tokens), max_len, dtype=torch.bool)

    # list of size BATCH_SZ of tensors of size (max_len)
    padded = [torch.zeros((1, max_len), dtype=int) for i in range(len(tgt_tokens))]
    for i in range(len(padded)):
        padded[i][:, : tgt_tokens[i].size(0)] = tgt_tokens[i]
        key_padding_mask[i][tgt_tokens[i].size(0) - 1 :] = 1

    key_padding_mask = key_padding_mask.unsqueeze(1)
    key_padding_mask = key_padding_mask.expand(-1, max_len, -1)

    cat = torch.concat(padded)

    tgt_embedding_fn = Embedding(len(tgt_vocab) + len(tgt_base) + 1, d_model)
    tgt_embeddings = tgt_embedding_fn(cat)
    return (tgt_embeddings, key_padding_mask)


# TODO(implement)
def decoder_unembed(Z_d):
    return Z_d


def simulate(t_e: List[str], t_d: List[str], h, num_layers) -> None:
    d_model = 200
    Z_e, key_padding_mask = tokenize_input(t_e, d_model)
    n_src = max([sentence.size(0) for sentence in Z_e])
    encoder = Encoder(n_src, h, d_model, 6, key_padding_mask)
    Z_e = encoder(Z_e)

    Z_d, key_padding_mask = tokenize_output(t_d, d_model)
    n_tgt = Z_d.size(0)
    n_tgt = max([sentence.size(0) for sentence in Z_d])
    decoder = Decoder(n_tgt, h, d_model, 6, Z_e, key_padding_mask)
    Z_d = decoder(Z_d)

    out = decoder_unembed(Z_d)
    return out


if __name__ == "__main__":
    de, en = [], []
    nrows = 50
    with open("./data/wmt14_translate_de-en_train.csv", encoding="utf-8") as f:
        data = pd.read_csv(
            "./data/wmt14_translate_de-en_train.csv", lineterminator="\n", nrows=nrows
        )
        for i, row in data.iterrows():
            de.append(row.iloc[0])
            en.append(row.iloc[1])

    # src_decoded = src_tokenizer.decode(src_tokens, src_base, src_vocab)
    # assert src_decoded == de
    # with open("./out/src_base.json", "w+") as fp:
    #     json.dump(src_base, fp)

    # with open("./out/src_vocab.json", "w+") as fp:
    #     json.dump(src_vocab, fp)

    # tgt_decoded = tgt_tokenizer.decode(tgt_tokens, tgt_base, tgt_vocab)
    # assert tgt_decoded == en
    # with open("./out/tgt_base.json", "w+") as fp:
    #     json.dump(tgt_base, fp)

    # with open("./out/tgt_vocab.json", "w+") as fp:
    #     json.dump(tgt_vocab, fp)

    out = simulate(de, en, 8, 6)
    print("[INFO] final output size: ", out.size())
