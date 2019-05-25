import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTG.module import AttentionLayer
from CTG.model import Decoder


class LSTMGumbelDecoder(Decoder):

    def __init__(self, args, embed, eos_index):
        super().__init__()
        self.layers = args['layers']
        self.embed_dim = args['embed_dim']
        self.source_hids_layer_norm = nn.LayerNorm(self.embed_dim * 2)
        self.rnn = nn.ModuleList([
            nn.LSTMCell(self.embed_dim, self.embed_dim)
            for _ in range(self.layers)
        ])
        self.rnn_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim)
            for _ in range(self.layers)
        ])
        self.attentions = nn.ModuleList([
            AttentionLayer(query_embed=self.embed_dim,
                           key_embed=self.embed_dim * 2,
                           val_embed=self.embed_dim * 2,
                           hidden_embed=self.embed_dim,
                           out_embed=self.embed_dim)
            for _ in range(self.layers)
        ]) if 'encoder_attention' in args and args['encoder_attention'] else [None] * self.layers
        self.atten_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embed_dim)
            for _ in range(self.layers)
        ]) if 'encoder_attention' in args and args['encoder_attention'] else [None] * self.layers
        self.tau = args['tau']
        self.use_tricky_rescale = args['use_tricky_rescale']
        self.output_layer_norm = nn.LayerNorm(self.embed_dim)
        self.embed = embed
        self.embed2word = embed.weight
        self.eos_index = eos_index
        self.maxlen = args['maxlen']
        self.dropout = args['dropout']

    def forward(self, var):
        source_hids, src_non_pad_mask = var['source_hids'], var['src_non_pad_mask']
        seed, temperature, seqlen = var['seeds'], var['temperature'], var['seqlen']

        temperature = max(temperature, self.tau)

        source_hids = self.source_hids_layer_norm(source_hids) / math.sqrt(self.embed_dim)

        bsz = seed.size(1)
        prev_states = [(seed[i], seed[i+self.layers]) for i in range(self.layers)]
        prev_output = torch.cat([self.embed2word[self.eos_index].unsqueeze(0)] * bsz, dim=0)
        output_logits = []
        for _ in range(seqlen):
            states = []
            for i, (rnn, rnn_ln, atten, atten_ln) in enumerate(zip(self.rnn,
                                                                   self.rnn_layer_norms,
                                                                   self.attentions,
                                                                   self.atten_layer_norms)):
                prev_state = rnn(prev_output, prev_states[i])
                output = F.dropout(prev_state[0], p=self.dropout)
                prev_output = prev_output + output
                prev_output = rnn_ln(prev_output) / math.sqrt(self.embed_dim)

                states.append(prev_state)

                if atten is not None:
                    atten, _ = atten(prev_output, source_hids, source_hids, src_non_pad_mask)
                    atten = F.dropout(atten, p=self.dropout)
                    prev_output = atten + prev_output
                    prev_output = atten_ln(prev_output) / math.sqrt(self.embed_dim)

            logits = F.linear(prev_output, self.embed2word)
            if self.use_tricky_rescale is not None:
                logits = TrickyRescale(temperature)(logits)
            else:
                logits = logits / temperature
            logits = F.log_softmax(logits, dim=-1)
            output_logits.append(logits.unsqueeze(1))
            # if self.training:
            probs = F.softmax(logits, dim=-1)
            prev_output = F.linear(probs, self.embed2word.transpose(0, 1))
            # else:
            #     indices = torch.argmax(logits, dim=-1, keepdim=False)
            #     prev_output = self.embed(indices)
            prev_states = states

        generation = torch.cat(output_logits, dim=1)

        var['generation'] = generation
        return var


class TrickyRescale(torch.autograd.Function):

    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x):
        return x / self.tau

    def backward(self, grad):
        return grad