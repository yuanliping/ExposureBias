from CTG.model import LSTMDecoder
from CTG.model import LSTMGumbelDecoder

import torch.nn as nn


class LSTMLanguageModel(nn.Module):

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    @classmethod
    def build_model(cls, dictionary, args):

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            embed = nn.Embedding(num_embeddings, embed_dim)
            nn.init.normal_(embed.weight, mean=0, std=embed_dim ** -0.5)
            return embed

        embed = build_embedding(dictionary, args['embed_dim'])
        decoder = LSTMDecoder(args, embed, dictionary.eos()).cuda()
        return LSTMLanguageModel(decoder)

    def forward(self, var):
        return self.language_model(var)
