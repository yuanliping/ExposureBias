import torch.nn as nn
import random


class EncDecDisc(nn.Module):

    def __init__(self, args, encoder, decoder, discriminator):
        super().__init__()
        self.style_num = args['style_num']
        assert self.encoder.out_dim % (self.style_num + 1) == 0, 'Encoder output dimension does not match style num'
        self.fdim = self.encoder.out_dim // (self.style_num + 1)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def _sample_seed(self, var):
        mean, logvar = var['mean'], var['logvar']
        noise = random.random() if self.training else 0
        var['seeds'] = mean + (logvar / 2).exp() * noise
        return var

    def _disentangle(self, var):
        pass

    def forward(self, var):
        var = self.encoder(var)
        var = self._sample_seed(var)
        var = self.decoder(var)
        var = self.discriminator(var)
        return var
