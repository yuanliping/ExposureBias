import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from CTG.module import AttentionLayer
from CTG.model import Decoder

import math


class LSTMWEaMDecoder(Decoder):

    def __init__(self, args, embed, eos_index):
        super().__init__()
        self.layers = args['layers']
        self.embed_dim = args['embed_dim']
        self.source_hids_layer_norm = nn.LayerNorm(self.embed_dim * 2)
        self.embed = embed
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
        self.output_layer_norm = nn.LayerNorm(self.embed_dim)
        self.embed = embed
        self.embed2word = embed.weight
        self.eos_index = eos_index
        self.maxlen = args['maxlen']
        self.dropout = args['dropout']
        self.gold_input = args['gold_input']
        self.residual_connection = args['residual_connection']
        self.mask_low_probs = args['mask_low_probs']
        self.min_margin, self.max_margin = eval(args['margin'])
        self.warmup_with_gold = args['warmup_with_gold']
        self.warmup_margin_with_gold = args['warmup_margin_with_gold']
        self.min_gold_margin_rate = args['min_gold_margin_rate']
        self.test_argmax = args['test_argmax']
        self.rbf = args['rbf']
        self.logit_anneal = args['logit_anneal']

    def forward(self, var):
        # noise = var['noise']
        subspace_count = 0
        source_hids, src_non_pad_mask = var['source_hids'], var['src_non_pad_mask']
        tgt_non_pad_mask = var['tgt_non_pad_mask']
        seed, temperature = var['seeds'], var['temperature']
        seqlen = var['seqlen'] if 'seqlen' in var else self.maxlen

        bsz = seed.size(1)

        margin = max(self.min_margin, self.max_margin * temperature)

        '''
        Get Gold Sentence
        '''
        gold, gold_embed = None, None
        if self.training:
            gold = var['tgt']
            gold_embed = self.embed(gold)

        '''
        Rescaling factor if logits requires annealing
        '''
        rescale = 1
        if self.logit_anneal:
            rescale = max(temperature, self.tau)

        '''
        Gating probability when warming up
        '''
        if not self.gold_input:
            # yuan_rate = max(temperature, 0.25)
            gold_sample = Variable(torch.cuda.FloatTensor([temperature]), requires_grad=False)
            gold_sample = torch.cat([gold_sample.view(1, 1)] * bsz, dim=0)
            gold_margin_rate = max(temperature, self.min_gold_margin_rate)
            gold_margin_sample = Variable(torch.cuda.FloatTensor([gold_margin_rate]), requires_grad=False)
            gold_margin_sample = torch.cat([gold_margin_sample.view(1, 1)] * bsz, dim=0)

        source_hids = self.source_hids_layer_norm(source_hids) / math.sqrt(self.embed_dim)
        prev_states = [(seed[i], seed[i + self.layers]) for i in range(self.layers)]
        prev_output = torch.cat([self.embed2word[self.eos_index].unsqueeze(0)] * bsz, dim=0)
        output_logits, generation_embed, gold_masks = [], [], []
        hidden_embed = []
        for l in range(seqlen):
            states = []
            '''
            Compute output of rnn
            '''
            for i, (rnn, rnn_ln, atten, atten_ln) in enumerate(zip(self.rnn,
                                                                   self.rnn_layer_norms,
                                                                   self.attentions,
                                                                   self.atten_layer_norms)):
                prev_state = rnn(prev_output, prev_states[i])
                output = F.dropout(prev_state[0], p=self.dropout, training=self.training)
                prev_output = prev_output + output
                prev_output = rnn_ln(prev_output) / math.sqrt(self.embed_dim)

                states.append(prev_state)

                if atten is not None:
                    atten, _ = atten(prev_output, source_hids, source_hids, src_non_pad_mask)
                    atten = F.dropout(atten, p=self.dropout, training=self.training)
                    prev_output = atten + prev_output
                    prev_output = atten_ln(prev_output) / math.sqrt(self.embed_dim)

            '''
            Logits by linear product
            '''
            hidden_embed.append(prev_output.unsqueeze(1))
            logits = F.linear(prev_output, self.embed2word)
            logits = logits / rescale
            '''
            Use RBF for logits
            '''
            if self.rbf:
                po_sqr = (prev_output ** 2).sum(dim=-1, keepdim=True)
                e2w_sqr = (self.embed2word ** 2).sum(dim=-1).unsqueeze(0)
                logits = 2 * logits - po_sqr - e2w_sqr

            '''
            Mask low probability
            '''
            if not self.gold_input and self.mask_low_probs:
                threshold, _ = logits.max(dim=-1, keepdim=True)
                threshold = threshold - margin
                if self.training:
                    gold_score = logits.gather(dim=-1, index=gold[:, l:l + 1])
                    if self.warmup_margin_with_gold:
                        gate = gold_margin_sample.bernoulli()
                        threshold = gate * (gold_score - margin) + (1 - gate) * threshold
                    gold_masks.append((threshold <= gold_score).float())
                below_threshold_mask = (logits < threshold).float()
                logits = logits + below_threshold_mask * -1e4

            '''
            Add log_probability to output
            '''
            logits = F.log_softmax(logits, dim=-1)
            output_logits.append(logits.unsqueeze(1))
            # logits = torch.exp(logits)
            if self.gold_input:
                '''
                Gold Input
                '''
                if self.training:
                    prev_output = gold_embed[:, l, :]
                else:
                    next_word = logits.argmax(dim=-1)
                    if l < var['tgt'].size(1):
                        subspace_count += torch.sum(next_word == var['tgt'][:, l])
                    prev_output = self.embed(next_word)
            else:
                '''
                Attentive Word Embedding as Input
                '''
                if self.training or not self.test_argmax:
                    probs = F.softmax(logits, dim=-1)
                    next_word_embed = F.linear(probs, self.embed2word.transpose(0, 1))
                    if self.residual_connection:
                        regressed_embed = (prev_output + next_word_embed) / math.sqrt(2.)
                    else:
                        regressed_embed = next_word_embed
                else:
                    max_index = logits.argmax(dim=-1)
                    regressed_embed = self.embed(max_index)

                if self.warmup_with_gold and self.training:
                    gate = gold_sample.bernoulli()
                    prev_output = gate * gold_embed[:, l, :] + (1 - gate) * regressed_embed
                    generation_embed.append(regressed_embed.unsqueeze(1))
                else:
                    prev_output = regressed_embed

            # prev_output = prev_output + torch.normal(mean=torch.zeros_like(prev_output), std=noise)
            prev_output = F.dropout(prev_output, p=self.dropout, training=self.training)
            prev_states = states

        hidden_embed = torch.cat(hidden_embed, dim=1)
        generation = torch.cat(output_logits, dim=1)
        generation_embed = torch.cat(generation_embed, dim=1) if not self.gold_input \
                                                                 and self.warmup_with_gold \
                                                                 and self.training else None
        gold_masks = torch.cat(gold_masks,
                               dim=1) if not self.gold_input and self.mask_low_probs and self.training else None
        tgt_non_pad_mask = tgt_non_pad_mask * gold_masks if gold_masks is not None else tgt_non_pad_mask

        var['hidden_embed'] = hidden_embed
        var['generation'] = generation
        var['gold_embed'] = gold_embed
        var['generation_embed'] = generation_embed
        var['tgt_non_pad_mask'] = tgt_non_pad_mask
        var['subspace_count'] = subspace_count
        return var
