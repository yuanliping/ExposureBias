from CTG.model import LSTMEncoder
from CTG.model import LSTMGumbelDecoder
from CTG.model import LSTMWEaMDecoder
from CTG.model import VAEEncDec

import torch.nn as nn


class GumbelVAE(VAEEncDec):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, dictionary, args):

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            embed = nn.Embedding(num_embeddings, embed_dim)
            nn.init.normal_(embed.weight, mean=0, std=embed_dim ** -0.5)
            return embed

        embed = build_embedding(dictionary, args['embed_dim'])
        encoder = LSTMEncoder(args, embed).cuda()
        decoder = LSTMGumbelDecoder(args, embed, dictionary.eos()).cuda()
        return GumbelVAE(encoder, decoder)


class WEaMVAE(VAEEncDec):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, dictionary, args):

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            embed = nn.Embedding(num_embeddings, embed_dim)
            nn.init.normal_(embed.weight, mean=0, std=embed_dim ** -0.5)
            return embed

        embed = build_embedding(dictionary, args['embed_dim'])
        encoder = LSTMEncoder(args, embed).cuda()
        decoder = LSTMWEaMDecoder(args, embed, dictionary.eos()).cuda()
        return WEaMVAE(encoder, decoder)
