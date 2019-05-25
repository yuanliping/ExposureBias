import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):

    def __init__(self, query_embed, key_embed, val_embed, hidden_embed, out_embed):
        super().__init__()

        self.scale = math.sqrt(query_embed)
        self.query_proj = nn.Linear(query_embed, hidden_embed, bias=False)
        self.key_proj = nn.Linear(key_embed, hidden_embed, bias=False)
        self.val_proj = nn.Linear(val_embed, hidden_embed, bias=False)
        self.output_proj = nn.Linear(hidden_embed, out_embed, bias=False)

    def forward(self, query, key, val, src_non_pad_mask=None):
        query = self.query_proj(query)
        key = self.key_proj(key)
        val = self.val_proj(val)

        attn_scores = torch.bmm(query.unsqueeze(1), key.transpose(-1, -2)) / self.scale

        if src_non_pad_mask is not None:
            src_pad_mask = 1 - src_non_pad_mask
            attn_scores += src_pad_mask.unsqueeze(1) * -1e4

        attn_scores = F.softmax(attn_scores, dim=-1)

        output = torch.bmm(attn_scores, val).squeeze(1)
        output = self.output_proj(output)

        return output, attn_scores
