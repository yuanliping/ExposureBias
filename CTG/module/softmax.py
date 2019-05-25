import torch.nn as nn
import torch.nn.functional as F


class LinearSoftmax(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature_in, feature_weight):
        logits = F.linear(feature_in, feature_weight)
        logits = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class RbfSoftmax(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature_in, feature_weight):
        ndim = len(feature_in.size())
        feature_in = feature_in.unsqueeze(ndim - 1)
        for _ in range(ndim - 1):
           feature_weight = feature_weight.unsqueeze(0)
        logits = -0.5 * (feature_in - feature_weight) ** 2
        logits = logits.sum(-1)
        logits = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        return probs, logits
