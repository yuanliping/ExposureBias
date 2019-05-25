from CTG.model import LSTMEncoder, TransformerEncoder
from CTG.model import LSTMWEaMDecoder, LSTMGumbelDecoder, TransformerWEaMDecoder
from CTG.model import EncDec, VAEEncDec

import torch.nn as nn


class WEaMTranslator(VAEEncDec):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, src_dict, tgt_dict, args):

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            embed = nn.Embedding(num_embeddings, embed_dim)
            nn.init.normal_(embed.weight, mean=0, std=embed_dim ** -0.5)
            return embed

        source_embed = build_embedding(src_dict, args['embed_dim'])
        if args['share_src_tgt_embed']:
            target_embed = source_embed
        else:
            target_embed = build_embedding(tgt_dict, args['embed_dim'])
        encoder = LSTMEncoder(args, source_embed).cuda()
        decoder = LSTMWEaMDecoder(args, target_embed, tgt_dict.eos()).cuda()
        return WEaMTranslator(encoder, decoder)


class GumbelTranslator(VAEEncDec):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, src_dict, tgt_dict, args):

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            embed = nn.Embedding(num_embeddings, embed_dim)
            nn.init.normal_(embed.weight, mean=0, std=embed_dim ** -0.5)
            return embed

        source_embed = build_embedding(src_dict, args['embed_dim'])
        target_embed = build_embedding(tgt_dict, args['embed_dim'])
        encoder = LSTMEncoder(args, source_embed).cuda()
        decoder = LSTMGumbelDecoder(args, target_embed, tgt_dict.eos()).cuda()
        return GumbelTranslator(encoder, decoder)


class TransformerWEaMTranslator(EncDec):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, src_dict, tgt_dict, args):

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            embed = nn.Embedding(num_embeddings, embed_dim)
            nn.init.normal_(embed.weight, mean=0, std=1)
            return embed

        source_embed = build_embedding(src_dict, args['embed_dim'])
        target_embed = build_embedding(tgt_dict, args['embed_dim'])
        encoder = TransformerEncoder(args, source_embed, src_dict.pad()).cuda()
        decoder = TransformerWEaMDecoder(args, target_embed, tgt_dict.eos()).cuda()
        return TransformerWEaMTranslator(encoder, decoder)