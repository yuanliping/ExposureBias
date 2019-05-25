import torch.nn as nn
import torch


class ReconstructionLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.eps = args['eps']

    def forward(self, generation, gold, non_pad_mask, hidden_embed=None, gold_embed=None):
        generation = generation.view(-1, generation.size(-1))
        gold, non_pad_mask = gold.view(-1), non_pad_mask.view(-1)
        nll_loss = -generation.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
        nll_loss = nll_loss * non_pad_mask
        smooth_loss = -generation.mean(dim=-1)
        smooth_loss = smooth_loss * non_pad_mask
        non_pad_num = non_pad_mask.sum()
        nll_loss = nll_loss.sum() / non_pad_num
        smooth_loss = smooth_loss.sum() / non_pad_num
        eps = self.eps if self.training else 0
        eps_i = eps / gold.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        if hidden_embed is not None:
            distance = torch.sum((hidden_embed - gold_embed) ** 2, dim=-1)
            mask_distance = non_pad_mask * distance.view(-1)
            avg_distance = torch.mean(mask_distance)
            loss += 0.01 * avg_distance
        return loss
