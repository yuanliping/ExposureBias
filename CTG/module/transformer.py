import torch.nn as nn
import torch.nn.functional as F

from CTG.module import MultiheadAttention


class Transformer(nn.Module):

    def __init__(self, embed_dim, head_num, dropout, enc_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(self.embed_dim, head_num)
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.enc_attn = MultiheadAttention(self.embed_dim, head_num) if enc_attn else None
        self.enc_ln = nn.LayerNorm(self.embed_dim) if enc_attn else None
        self.dropout = dropout
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim * 2)
        self.fc2 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x, mem_x, x_mask=None, enc=None, enc_mask=None):
        residual = x
        x = self.self_attn(query=x, key=mem_x, val=mem_x, mask=x_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.ln1(x)

        if self.enc_attn is not None:
            residual = x
            x = self.enc_attn(query=x, key=enc, val=enc, mask=enc_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.enc_ln(x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.ln2(x)
        return x
