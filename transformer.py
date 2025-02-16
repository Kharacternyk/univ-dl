from math import cos, sin, sqrt

from torch import nn
import torch


class Transformer(nn.Module):
    def __init__(
        self,
        in_vocabulary_size,
        out_vocabulary_size,
        sequence_length,
        depth,
        breadth,
        dropout,
        n_heads,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.in_embed = nn.Embedding(in_vocabulary_size, breadth)
        self.out_embed = nn.Embedding(out_vocabulary_size, breadth)
        self.pos_embed = PosEmbed(sequence_length, breadth)
        self.encoder = Encoder(depth, breadth, dropout, n_heads)
        self.decoder = Decoder(depth, breadth, dropout, n_heads)
        self.last = nn.Sequential(
            nn.Linear(breadth, out_vocabulary_size),
            nn.LogSoftmax(-1),
        )

    def forward(self, input, output, e_mask, d_mask):
        input = self.pos_embed(self.in_embed(input))
        output = self.pos_embed(self.out_embed(output))
        return self.last(
            self.decoder(
                output,
                self.encoder(input, e_mask),
                e_mask,
                d_mask,
            )
        )


class Encoder(nn.Module):
    def __init__(self, depth, breadth, dropout, n_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(breadth, dropout, n_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm([breadth])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, depth, breadth, dropout, n_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(breadth, dropout, n_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm([breadth])

    def forward(self, x, e, e_mask, mask):
        for layer in self.layers:
            x = layer(x, e, e_mask, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, breadth, dropout, n_heads):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm([breadth]), nn.LayerNorm([breadth])])
        self.drops = nn.ModuleList([nn.Dropout(dropout), nn.Dropout(dropout)])
        self.attn = MultiheadAttention(breadth, n_heads, dropout)
        self.ff = FFLayer(breadth, dropout)

    def forward(self, x, mask):
        xs = self.norms[0](x), self.norms[1](x)
        x = x + self.drops[0](self.attn(xs[0], xs[0], xs[0], mask))
        x = x + self.drops[1](self.ff(xs[1]))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, breadth, dropout, n_heads):
        super().__init__()
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm([breadth]),
                nn.LayerNorm([breadth]),
                nn.LayerNorm([breadth]),
            ]
        )
        self.drops = nn.ModuleList(
            [nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)]
        )
        self.attns = nn.ModuleList(
            [
                MultiheadAttention(breadth, n_heads, dropout),
                MultiheadAttention(breadth, n_heads, dropout),
            ]
        )
        self.ff = FFLayer(breadth, dropout)

    def forward(self, x, e, e_mask, mask):
        xs = self.norms[0](x), self.norms[1](x), self.norms[2](x)
        x = x + self.drops[0](self.attns[0](xs[0], xs[0], xs[0], mask))
        x = x + self.drops[1](self.attns[1](xs[1], e, e, e_mask))
        x = x + self.drops[2](self.ff(xs[2]))
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, breadth, n_heads, dropout):
        super().__init__()
        self.w_q = nn.Linear(breadth, breadth)
        self.w_k = nn.Linear(breadth, breadth)
        self.w_v = nn.Linear(breadth, breadth)
        self.w_0 = nn.Linear(breadth, breadth)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.n_heads = n_heads
        self.breadth = breadth
        self.d = breadth // n_heads

    def forward(self, q, k, v, mask):
        length = len(q)
        q = self._reshape(self.w_q(q))
        k = self._reshape(self.w_k(k))
        v = self._reshape(self.w_v(v))
        scores = self.self_attention(q, k, v, mask)
        return self.w_0(
            scores.transpose(1, 2).contiguous().view(length, -1, self.breadth)
        )

    def self_attention(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.d)
        scores.masked_fill_(mask.unsqueeze(1) == 0, -1e9)
        scores = self.drop(self.softmax(scores))
        return torch.matmul(scores, v)

    def _reshape(self, x):
        return x.view(len(x), -1, self.n_heads, self.d).transpose(1, 2)


class PosEmbed(nn.Module):
    def __init__(self, sequence_length, breadth):
        super().__init__()
        matrix = torch.zeros(sequence_length, breadth).cuda()
        for j in range(matrix.size(0)):
            for i in range(matrix.size(1)):
                fun = cos if i % 2 else sin
                matrix[j, i] = fun(j / (10000 ** (i / breadth * 2)))
        self.matrix = matrix.unsqueeze(0).detach()
        self.factor = sqrt(breadth)

    def forward(self, x):
        return x * self.factor + self.matrix


class FFLayer(nn.Sequential):
    def __init__(self, breadth, dropout):
        super().__init__(
            nn.Linear(breadth, breadth * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(breadth * 4, breadth),
        )
