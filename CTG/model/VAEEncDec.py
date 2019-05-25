from CTG.model import EncDec

import torch


class VAEEncDec(EncDec):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def _sample_seed(self, var):
        mean, logvar = var['mean'], var['logvar']
        if logvar is not None:
            noise = torch.randn(mean.size()).cuda() if self.training else 0
            var['seeds'] = mean + (logvar / 2).exp() * noise
        else:
            var['seeds'] = mean
        return var

    def forward(self, var):
        var = self.encoder(var)
        var = self._sample_seed(var)
        var = self.decoder(var)
        return var
