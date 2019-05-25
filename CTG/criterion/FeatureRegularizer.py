import torch.nn as nn


class FeatureRegularizer(nn.Module):

    def __init__(self, dictionary, args):
        super().__init__()
        self.pad = dictionary.pad()
        self.beta = args['beta']

    def forward(self, gold_embed, predict_embed, non_pad_mask):
        dist = ((gold_embed - predict_embed) ** 2).sum(dim=-1)
        reg = non_pad_mask * dist
        return reg.sum() * self.beta
