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
    X_lengths = (target[:,:,0].cpu().detach().numpy() == -1).argmax(1)

    mse_sum = 0
    batch_size = target.shape[0]
    # loop over batch size
    for i in range(batch_size):
        y = target[i, :X_lengths[i], :]
        y_hat = output[i, :X_lengths[i], :]
        mse_sum += F.mse_loss(y, y_hat)

    return mse_sum / batch_size 
