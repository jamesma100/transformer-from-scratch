from tokenizer import Tokenizer
from typing import List, Dict
import csv
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

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

    def weight(self):
        return self.embedding.weight

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        return self.embedding(X) * math.sqrt(self.d_model)


class Unembedding(nn.Module):
    def __init__(self, vocab_sz, d_model):
        super(Unembedding, self).__init__()
        self.d_model = d_model
        self.unembedding = nn.Linear(d_model, vocab_sz)

    # Apparently it is beneficial to tie the weights here to the transpose of
    # the weights of the embedding layer
    #
    # See https://paperswithcode.com/paper/using-the-output-embedding-to-improve
    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, VOCAB_SZ)
        """
        return F.softmax(self.unembedding(X), dim=-1)


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
        return X + self.dropout(X + self.PE[: X.size(-2), : X.size(-1)])


class LayerNorm(nn.Module):
    def __init__(self, n, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.d_model = d_model
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
        return (X - mean) / torch.sqrt(var + self.eps) * self.alpha[
            : X.size(-2), :self.d_model
        ] + self.beta[: X.size(-2), :self.d_model]


class ResidualConnection(nn.Module):
    def __init__(self, p_drop=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, X, X_normalized, sublayer):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)
        X_normalized: tensor of size (BATCH_SZ, N, D_MODEL)
        sublayer: function mapping tensor -> tensor, both of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        return X + self.dropout(sublayer(X_normalized))


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
        X_normalized = self.norm(X)
        return self.add(X, X_normalized, sublayer)


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


class MultiheadedAttention(nn.Module):
    def __init__(self, n, h, d_model, att_mask=False):
        """
        Args:
        att_mask: whether or not illegal connections should be masked
            This should be set to True in the decoder self-attention block
        """
        super(MultiheadedAttention, self).__init__()
        self.n = n
        self.h = h
        self.d_q = d_model // h
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.queries = nn.ModuleList([nn.Linear(d_model, self.d_q) for i in range(h)])
        self.keys = nn.ModuleList([nn.Linear(d_model, self.d_k) for i in range(h)])
        self.values = nn.ModuleList([nn.Linear(d_model, self.d_v) for i in range(h)])
        self.att_mask = att_mask

    def attention(self, Q, K, V):
        """
        Args:
        Q: tensor of size (BATCH_SZ, N, D_HEAD)
        K: tensor of size (BATCH_SZ, N, D_HEAD)
        V: tensor of size (BATCH_SZ, N, D_HEAD)

        Output: tensor of size (BATCH_SZ, N, D_HEAD)
        """
        QKT = Q.matmul(K.transpose(-2, -1))
        if self.att_mask:
            indices = torch.triu_indices(Q.size(1), Q.size(1), offset=1)
            if QKT.dim() == 3:
                QKT[:, indices[0], indices[1]] = float("-inf")
            elif QKT.dim() == 2:
                QKT[indices[0], indices[1]] = float("-inf")

        return (QKT / math.sqrt(self.d_k)).softmax(dim=-1).matmul(V)

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
        res = torch.cat(heads, dim=-1)
        return res


class EncoderLayer(nn.Module):
    def __init__(self, n, h, d_model):
        super(EncoderLayer, self).__init__()
        self.layer1 = MultiheadedAttention(n, h, d_model)
        self.add_and_norm1 = AddAndNorm(n, d_model)
        self.layer2 = FFN(n, d_model, d_model * 4)
        self.add_and_norm2 = AddAndNorm(n, d_model)

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        attention_out = self.add_and_norm1(X, lambda x: self.layer1(x, x, x))
        ffn_out = self.add_and_norm2(attention_out, lambda x: self.layer2(x))
        return ffn_out


class DecoderLayer(nn.Module):
    def __init__(self, n, h, d_model):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)
            Represents the output of the encoder
        """
        super(DecoderLayer, self).__init__()

        self.layer1 = MultiheadedAttention(n, h, d_model, att_mask=True)
        self.add_and_norm1 = AddAndNorm(n, d_model)
        self.layer2 = MultiheadedAttention(n, h, d_model)  # cross attention
        self.add_and_norm2 = AddAndNorm(n, d_model)
        self.layer3 = FFN(n, d_model, d_model * 4)
        self.add_and_norm3 = AddAndNorm(n, d_model)

    def forward(self, X, Y):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)
        Y: tensor of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        self_attention_out = self.add_and_norm1(Y, lambda x: self.layer1(x, x, x))
        cross_attention_out = self.add_and_norm2(
            self_attention_out, lambda x: self.layer2(x, X, X)
        )
        ffn_out = self.add_and_norm3(cross_attention_out, lambda x: self.layer3(x))
        return ffn_out


class Encoder(nn.Module):
    def __init__(self, n, h, d_model, num_layers):
        super(Encoder, self).__init__()
        self.pe = PosEnc(n, d_model)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(n, h, d_model) for i in range(num_layers)]
        )

    def forward(self, X):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        X = self.pe(X)
        for encoder_layer in self.encoder_layers:
            X = encoder_layer(X)
        return X


class Decoder(nn.Module):
    def __init__(self, n, h, d_model, num_layers, vocab_sz):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)
            Represents the output of the encoder
        """
        super(Decoder, self).__init__()
        self.pe = PosEnc(n, d_model)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(n, h, d_model) for i in range(num_layers)]
        )
        self.unembedding = Unembedding(vocab_sz, d_model)

    def forward(self, X, Y=None):
        """
        Args:
        X: tensor of size (BATCH_SZ, N, D_MODEL)
        Y: tensor of size (BATCH_SZ, N, D_MODEL)

        Output: tensor of size (BATCH_SZ, N, D_MODEL)
        """
        Y = X.clone() if Y is None else self.pe(Y)
        for decoder_layer in self.decoder_layers:
            Y = decoder_layer(X, Y)
        return self.unembedding(Y)


class Transformer(nn.Module):
    def __init__(self, n, h, d_model, num_layers, vocab_sz):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n, h, d_model, num_layers)
        self.decoder = Decoder(n, h, d_model, num_layers, vocab_sz)
        self.tokenizer = Tokenizer("")
        self.vocab_sz = vocab_sz
        self.d_model = d_model
        self.embedding = Embedding(vocab_sz, d_model)

    def forward(self, X, Y=None):
        X_embedding = self.embedding(X)
        Y_embedding = self.embedding(Y) if Y is not None else Y
        X_embedding = self.encoder(X_embedding)
        return self.decoder(X_embedding, Y_embedding)

    def generate(self, cur_tokens, seq_len, vocab_sz, d_model, blk_sz):
        for _ in range(seq_len):
            if cur_tokens.size(1) > blk_sz:
                cur_tokens = cur_tokens[:, -blk_sz:]
            Y_hat = self(cur_tokens)

            probs = F.softmax(Y_hat, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            print(
                self.tokenizer.decode([next_token[0][0].item()], base, vocab),
                end="",
                flush=True,
            )
            cur_tokens = torch.cat((cur_tokens, next_token), dim=1)
        print("")


def get_batch(src_tokens, blk_sz, batch_sz):
    idx = torch.randint(len(src_tokens) - blk_sz, (batch_sz,))
    src_tokens = torch.tensor(src_tokens)
    X = torch.stack([src_tokens[i : i + blk_sz] for i in idx])
    Y = torch.stack([src_tokens[i + 1 : i + blk_sz + 1] for i in idx])
    return (X, Y)


def get_data(src_tokens, blk_sz):
    sz = len(src_tokens)
    src_tokens = torch.tensor(src_tokens)
    X = torch.stack([src_tokens[i : i + blk_sz] for i in range(sz - blk_sz - 1)])
    Y = torch.stack([src_tokens[i + 1 : i + blk_sz + 1] for i in range(sz - blk_sz)])
    return (X, Y)


def get_loss(Y_hat, Y):
    return F.cross_entropy(Y_hat, Y)
