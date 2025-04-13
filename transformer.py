import torch
import torch.nn as nn
import math

# TODO(implement)
def encoder_tokenize(t_e):
    return torch.randn(100, 512)

# TODO(implement)
def decoder_tokenize(t_d):
    return torch.randn(100, 512)

# TODO(implement)
def encoder_final_layer_norm(Z_e):
    return Z_e

# TODO(implement)
def decoder_final_layer_norm(Z_d):
    return Z_d

# TODO(implement)
def decoder_unembed(Z_d):
    return Z_d


class PosEnc(nn.Module):
    def __init__(self, n, d_model, base=10000):
        super(PosEnc, self).__init__()
        # TODO: make this not dumb
        self.PE = torch.zeros(n, d_model)
        for pos in range(n):
            for i in range(d_model // 2):
                denom = math.pow(10000, 2*i/d_model)
                self.PE[pos][2*i] = math.sin(pos / denom)
                self.PE[pos][2*i+1] = math.cos(pos / denom)
    def forward(self, X):
        return X + self.PE

class FFN(nn.Module):
    """
    y = xA' + b
    NOTE: with nn.Linear you provide the dimensions of A' not A! but the
    weight will have dimension of A.

    Input: X(nx512), W1(512x2048), W2(2048x512), b1(1x2048), b2(1x512)
    Output: max{0, (xW1 + b1)}W2 + b2 (nx512)
    """
    def __init__(self, n, d_model, dff):
        super(FFN, self).__init__()
        self.W1 = nn.Linear(d_model, dff)
        self.W2 = nn.Linear(dff, d_model)

    def forward(self, X):
        return self.W2(self.W1(X).relu())

class MultiheadedAttention(nn.Module):
    def __init__(self, n, h, d_model):
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

    def attention(self, Q, K, V):
        sm = torch.nn.Softmax(dim=1)
        QKT = torch.matmul(Q, K.transpose(0, 1))
        return torch.matmul(sm(QKT / math.sqrt(self.d_k)), V)

    # NOTE: X_Q = X_K = X_V if self-attention
    def forward(self, X_Q, X_K, X_V):
        Qs = [self.queries[i](X_Q) for i in range(self.h)]
        Ks = [self.keys[i](X_K) for i in range(self.h)]
        Vs = [self.values[i](X_V) for i in range(self.h)]
        heads = [self.attention(Q, K, V) for Q, K, V in zip(Qs, Ks, Vs)]
        return torch.cat(heads, dim=1)

class ResidualConnection(nn.Module):
    def __init__(self):
        super(ResidualConnection, self).__init__()

    def forward(self, X, sublayer):
        return X + sublayer(X)

class EncoderLayer(nn.Module):
    def __init__(self, n, h, d_model):
        super(EncoderLayer, self).__init__()
        self.n = n
        self.h = h
        self.d_model = d_model
        self.layer1 = MultiheadedAttention(n, h, d_model)
        self.res1 = ResidualConnection()
        self.layer2 = FFN(n, d_model, d_model * 4)
        self.res2 = ResidualConnection()

    def forward(self, X):
        attention_out = self.res1(X, lambda x: self.layer1(x, x, x))
        ffn_out = self.res2(attention_out, lambda x: self.layer2(x))
        return ffn_out

class DecoderLayer(nn.Module):
    def __init__(self, n, h, d_model, Z_e):
        super(DecoderLayer, self).__init__()
        self.n = n
        self.h = h
        self.d_model = d_model
        self.Z_e = Z_e

        self.layer1 = MultiheadedAttention(n, h, d_model)
        self.res1 = ResidualConnection()
        self.layer2 = MultiheadedAttention(n, h, d_model)
        self.res2 = ResidualConnection()
        self.layer3 = FFN(n, d_model, d_model * 4)
        self.res3 = ResidualConnection()

    def forward(self, X):
        self_attention_out = self.res1(X, lambda x: self.layer1(x, x, x))
        cross_attention_out = self.res2(self_attention_out, lambda x: self.layer2(self.Z_e, x, x))
        ffn_out = self.res3(cross_attention_out, lambda x: self.layer3(x))
        return ffn_out

def simulate(t_e, t_d, n, d_model, h, num_layers):
    Z_e = encoder_tokenize(t_d)
    pe = PosEnc(n, d_model)
    Z_e = pe(Z_e)
    encoder_layers = [EncoderLayer(n, h, d_model) for i in range(num_layers)]
    for encoder_layer in encoder_layers:
        Z_e = encoder_layer(Z_e)
    Z_e = encoder_final_layer_norm(Z_e)
    # <Encoder end>
    
    # <Decoder start>
    Z_d = decoder_tokenize(t_d)
    pe = PosEnc(n, d_model)
    Z_d = pe(Z_d)
    decoder_layers = [DecoderLayer(n, h, d_model, Z_e) for i in range(num_layers)]
    for decoder_layer in decoder_layers:
        Z_d = decoder_layer(Z_d)
    Z_d = decoder_final_layer_norm(Z_d)
    # <Decoder end>
    
    out = decoder_unembed(Z_d)
    return out

if __name__ == "__main__":
    t_e = ["hello", "everyone"]
    t_d = ["hola", "todos"]
    out = simulate(t_e, t_d, 100, 512, 8, 6)
    print("[DEBUG] output: \n", out)
