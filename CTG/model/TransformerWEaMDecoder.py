import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from CTG.model import Decoder
from CTG.module import Transformer


class TransformerWEaMDecoder(Decoder):

    def __init__(self, args, embed, eos_index):
        super().__init__()
        self.layers = args['layers']
        self.embed_dim = args['embed_dim']
        self.dropout = args['dropout']
        self.tau = args['tau']
        self.embed = embed
        self.transformer = nn.ModuleList([
            Transformer(self.embed_dim, args['head_num'], self.dropout, enc_attn=True)
            for _ in range(self.layers)
        ])
        self.eos_index = eos_index
        self.embed2word = embed.weight
        self.maxlen = args['maxlen']
        self.gold_input = args['gold_input']
        self.residual_connection = args['residual_connection']
        self.mask_low_probs = args['mask_low_probs']
        self.margin = args['margin']
        self.warmup_with_gold = args['warmup_with_gold']
        self.argmax = args['argmax']

    def forward(self, var):
        source_hids, src_pad_mask = var['src_hids'], var['src_pad_mask']
        temperature = var['temperature']
        seqlen = var['seqlen'] if 'seqlen' in var else self.maxlen

        gold, gold_embed = None, None
        if self.training:
            gold = var['tgt']
            gold_embed = self.embed(gold)

        rescale = max(temperature, self.tau)

        bsz = source_hids.size(0)
        x = torch.cat([self.embed2word[self.eos_index].unsqueeze(0)] * bsz, dim=0).unsqueeze(1)
        prev_embed = [None for _ in range(self.layers + 1)]
        prev_embed[0] = x
        output_logits = []
        for l in range(seqlen):
            for i, layer in enumerate(self.transformer):
                x = layer(x, prev_embed[i], x_mask=None, enc=source_hids, enc_mask=src_pad_mask)
                prev_embed[i + 1] = torch.cat([prev_embed[i + 1], x], dim=1) if prev_embed[i + 1] is not None else x
            logits = F.linear(x, self.embed2word)
            logits = logits / rescale

            logits = F.log_softmax(logits, dim=-1)
            output_logits.append(logits)

            if self.gold_input:
                '''
                Gold Input
                '''
                if self.training:
                    next_word_embed = gold_embed[:, l:l+1, :]
                else:
                    next_word = logits.argmax(dim=-1)
                    next_word_embed = self.embed(next_word)
            else:
                probs = F.softmax(logits, dim=-1)
                next_word_embed = F.linear(probs, self.embed2word.transpose(0, 1))
                if self.residual_connection:
                    next_word_embed = (x + next_word_embed) / math.sqrt(2.)
                else:
                    next_word_embed = next_word_embed

            x = F.dropout(next_word_embed, p=self.training)

        generation = torch.cat(output_logits, dim=1)
        var['generation'] = generation
        return var
