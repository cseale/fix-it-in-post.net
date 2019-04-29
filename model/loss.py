import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def squared_loss(output, target):
    return F.mse_loss(output, target)
