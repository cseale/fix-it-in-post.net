import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def squared_loss(output, target):
    return F.mse_loss(output, target)

def padded_loss(output, target):
    """
    output dimensions: batch x M x 129
    target dimensions: batch x N x 129, where N <= M
    The loss must NOT be computed on entries corresponding to indices which are greater than N
    """
    filtered_output = output[:target.shape[0], :target.shape[1], :target.shape[2]]
    return F.mse_loss(filtered_output, target)
