import torch
import torch.nn as nn
import torch.nn.functional as F

from CTG.model import Decoder

from CTG.module import LinearSoftmax, RbfSoftmax

class LSTMDecoder(Decoder):

    def __init__(self, args, embed, eos_index):
        super().__init__()
        self.embed = embed
        self.layers = args['layers']
        self.embed_dim = args['embed_dim']
        self.rnn = nn.ModuleList([
            nn.LSTMCell(self.embed_dim, self.embed_dim)
            for _ in range(self.layers)
        ])
        self.embed2word = embed.weight
        self.dropout = args['dropout']
        self.eos_idx = eos_index
        self.maxlen = args['maxlen']
        self.softmax = RbfSoftmax() if args['rbf_softmax'] else LinearSoftmax()

    def forward(self, var):
        prev_output = var['sent']
        bsz, seqlen = prev_output.size()
        prev_output_embed = self.embed(prev_output)
        prev_states = [None] * len(self.rnn)
        eos_embed = self.embed2word[self.eos_idx]
        eos_embed_pad = torch.cat([eos_embed.unsqueeze(0)] * bsz, dim=0).unsqueeze(1)
        prev_output_embed = torch.cat([eos_embed_pad, prev_output_embed], dim=1)
        prev_output_embed = prev_output_embed.transpose(0, 1)
        rnn_out = []
        for i in range(seqlen + 1):
            rnn_in = prev_output_embed[i]
            states = []
            for j, rnn in enumerate(self.rnn):
                residual = rnn_in
                state = rnn(rnn_in, prev_states[j])
                rnn_in = state[0]
                rnn_in = F.dropout(rnn_in, self.dropout, training=self.training)
                rnn_in = rnn_in + residual
                states.append(state)
            prev_states = states
            rnn_out.append(rnn_in.unsqueeze(1))
        rnn_out = torch.cat(rnn_out, dim=1)
        probs, logits = self.softmax(rnn_out, self.embed2word)

        var['generation'] = logits
        return var