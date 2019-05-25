import torch.nn as nn


class KLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, mean, logvar):
        return -(1 + logvar - logvar.exp() - mean ** 2).sum(dim=-1).mean() * 0.5