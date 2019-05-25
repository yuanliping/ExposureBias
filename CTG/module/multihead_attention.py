import math

import torch
from torch import nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, val, mask=None):
        bsz, qlen, _ = query.size()
        klen = key.size(1)
        q = self.linear(query, self.q_linear, bsz, qlen, transpose=False)
        k = self.linear(key, self.k_linear, bsz, klen, transpose=True)
        v = self.linear(val, self.v_linear, bsz, klen, transpose=False)
        logits = torch.bmm(q, k) / math.sqrt(self.embed_dim)
        if mask is not None:
            mask = mask.float()
            mask = torch.cat([mask.unsqueeze(1)] * self.num_head, dim=1).view(bsz * self.num_head, 1, klen)
            logits = logits - mask * 1e4
        probs = F.softmax(logits, dim=-1)
        out = torch.bmm(probs, v)
        out = out.view(bsz, self.num_head, qlen, self.head_dim).transpose(1, 2).contiguous().view(bsz, qlen, self.embed_dim)
        out = self.out_linear(out)
        return out

    def linear(self, x, x_linear, bsz, xlen, transpose=False):
        x = x_linear(x)
        x = x.view(bsz, xlen, self.num_head, self.head_dim)
        x = x.transpose(1, 2)
        if transpose:
            x = x.transpose(2, 3)
            x = x.contiguous()
            x = x.view(bsz * self.num_head, self.head_dim, xlen)
        else:
            x = x.contiguous()
            x = x.view(bsz * self.num_head, xlen, self.head_dim)
        return x

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

# import torch
# from torch import nn
# from torch.nn import Parameter
# import torch.nn.functional as F
#
#
# class MultiheadAttention(nn.Module):
#     """Multi-headed attention.
#
#     See "Attention Is All You Need" for more details.
#     """
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
#         self.scaling = self.head_dim**-0.5
#         self._mask = None
#
#         self.in_proj_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim))
#         if bias:
#             self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.in_proj_weight)
#         nn.init.xavier_uniform_(self.out_proj.weight)
#         if self.in_proj_bias is not None:
#             nn.init.constant_(self.in_proj_bias, 0.)
#             nn.init.constant_(self.out_proj.bias, 0.)
#
#     def forward(self, query, key, val, mask=None):
#
#         qkv_same = query.data_ptr() == key.data_ptr() == val.data_ptr()
#         kv_same = key.data_ptr() == val.data_ptr()
#
#         tgt_len, bsz, embed_dim = query.size()
#         assert embed_dim == self.embed_dim
#         assert list(query.size()) == [tgt_len, bsz, embed_dim]
#         assert key.size() == val.size()
#
#         if qkv_same:
#             q, k, v = self.in_proj_qkv(query)
#         elif kv_same:
#             q = self.in_proj_q(query)
#             if key is None:
#                 assert val is None
#                 k = v = q.new(0)
#             else:
#                 k, v = self.in_proj_kv(key)
#         else:
#             q = self.in_proj_q(query)
#             k = self.in_proj_k(key)
#             v = self.in_proj_v(val)
#         q *= self.scaling
#
#         src_len = k.size(0)
#
#         if mask is not None:
#             assert mask.size(0) == bsz
#             assert mask.size(1) == src_len
#
#         q = q.contiguous().view(tgt_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
#         k = k.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
#         v = v.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
#
#         attn_weights = torch.bmm(q, k.transpose(1, 2))
#         assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
#
#         if mask is not None:
#             # don't attend to padding symbols
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             attn_weights = attn_weights.float().masked_fill(
#                 mask.unsqueeze(1).unsqueeze(2),
#                 float('-inf'),
#             ).type_as(attn_weights)  # FP16 support: cast to float and back
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
#         attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
#         attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
#
#         attn = torch.bmm(attn_weights, v)
#         assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
#         attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#         attn = self.out_proj(attn)
#
#         return attn
#
#     def in_proj_qkv(self, query):
#         return self._in_proj(query).chunk(3, dim=-1)
#
#     def in_proj_kv(self, key):
#         return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)
#
#     def in_proj_q(self, query):
#         return self._in_proj(query, end=self.embed_dim)
#
#     def in_proj_k(self, key):
#         return self._in_proj(key, start=self.embed_dim, end=2*self.embed_dim)
#
#     def in_proj_v(self, value):
#         return self._in_proj(value, start=2*self.embed_dim)
#
#     def _in_proj(self, input, start=0, end=None):
#         weight = self.in_proj_weight
#         bias = self.in_proj_bias
#         weight = weight[start:end, :]
#         if bias is not None:
#             bias = bias[start:end]
#         return F.linear(input, weight, bias)
