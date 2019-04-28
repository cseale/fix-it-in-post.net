import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

"""
Squared Loss (TODO)

Calculated the squared loss between output and target values
"""
def squared_loss(output, target):
    return None
