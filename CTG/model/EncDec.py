import torch.nn as nn


class EncDec(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, var):
        var = self.encoder(var)
        var = self.decoder(var)
        return var
