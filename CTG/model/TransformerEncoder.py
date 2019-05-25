from CTG.model import Encoder
from CTG.module import Transformer

import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(Encoder):
    
    def __init__(self, args, embed, pad_idx):
        super().__init__()
        self.embed = embed
        self.layers = args['layers']
        self.embed_dim = args['embed_dim']
        self.dropout = args['dropout']
        self.pad_idx = pad_idx
        assert self.embed_dim == embed.weight.size(1)

        self.transformer = nn.ModuleList([
            Transformer(self.embed_dim, args['head_num'], self.dropout)
            for _ in range(self.layers)
        ])
    
    def forward(self, var):
        src, src_length = var['src'], var['src_lengths']
        src_embed = self.embed(src)
        src_embed = F.dropout(src_embed, p=self.dropout)
        pad_mask = src.eq(self.pad_idx)

        x = src_embed
        for layer in self.transformer:
            x = layer(x, x, pad_mask)

        var['src_hids'] = x
        var['src_pad_mask'] = pad_mask
        return var
