from CTG.model import Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(Encoder):
    
    def __init__(self, args, embed):
        super().__init__()
        self.embed = embed
        self.layers = args['layers']
        self.embed_dim = args['embed_dim']
        self.out_dim = args['encoder_out_dim']
        self.dropout = args['dropout']
        assert self.embed_dim == embed.weight.size(1)

        self.rnn = nn.LSTM(self.embed_dim, self.embed_dim,
                           num_layers=self.layers,
                           dropout=self.dropout,
                           bidirectional=True,)
        self.ffn_mean = nn.Linear(self.embed_dim * 2, self.out_dim)
        self.ffn_logvar = nn.Linear(self.embed_dim * 2, self.out_dim) if args['vae'] else None
    
    def forward(self, var):
        src, src_length = var['src'], var['src_lengths']
        src_embed = self.embed(src)
        src_embed = F.dropout(src_embed, p=self.dropout, training=self.training)

        src_embed = src_embed.transpose(0, 1)
        src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, src_length.data.tolist())
        source_hids, (h, c) = self.rnn(src_embed)
        source_hids, _ = nn.utils.rnn.pad_packed_sequence(source_hids, padding_value=0.)
        source_hids = source_hids.transpose(0, 1)

        h, c = self.reshape_final_cell(h), self.reshape_final_cell(c)

        x = torch.cat([h, c], dim=0)

        mean = self.ffn_mean(x)
        logvar = self.ffn_logvar(x) if self.ffn_logvar is not None else None
        var['mean'] = mean
        var['logvar'] = logvar
        var['source_hids'] = source_hids
        return var

    def reshape_final_cell(self, x):
        bsz = x.size(1)
        x = x.view(self.layers, 2, bsz, -1)
        x = x.transpose(1, 2).contiguous()
        x = x.view(self.layers, bsz, -1)
        return x

