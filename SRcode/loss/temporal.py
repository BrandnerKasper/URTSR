import torch
import torch.nn as nn
from torch.nn import functional as F


# Code copied from NSDR https://github.com/Riga2/NSRD/blob/main/src/loss/temporal_loss.py
class TemporalLoss(nn.Module):
    def __init__(self):
        super(TemporalLoss, self).__init__()

    def forward(self, sr_pre, sr_cur):
        return F.smooth_l1_loss(sr_pre, sr_cur)
