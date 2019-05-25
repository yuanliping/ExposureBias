import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, var):
        raise NotImplementedError
