import torch.nn as nn
import torch.nn.functional as F


class OthogonalNeighborsRegularization(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.tau = args['tau']

    def forward(self, generation, non_pad_mask):
        generation = F.softmax(generation, dim=-1)
        generation = generation ** self.tau
        generation = generation / generation.sum(dim=-1, keepdim=True)
        left, right = generation[:, :-1, :], generation[:, 1:, :]
        reg = (left * right).sum(-1)
        non_pad_mask = non_pad_mask[:, 1:]
        reg = reg * non_pad_mask
        reg = reg.sum() / non_pad_mask.sum()
        return reg
