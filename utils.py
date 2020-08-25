import math
import torch.nn as nn
import random
import torch
from torch.distributions import Independent, Normal

  
from torch.autograd import Function

# Credit: https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/functional.py
class ReverseGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


reverse_grad = ReverseGrad.apply

def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


def mean_logits(logits, dim=0):
    """Return the mean logit, where the average is taken over across `dim` in probability space."""
    # p = e^logit / (sum e^logit) => log(p) = logit - log_sum_exp(logit)
    log_prob = logits - torch.logsumexp(logits, -1, keepdim=True)
    # mean(p) = 1/Z sum exp(log(p)) => log(mean(p)) = log_sum_exp(log(p)) - log(Z)
    log_mean_prob = torch.logsumexp(log_prob, 0) - math.log(log_prob.size(0))
    return log_mean_prob


def permute_idcs(n_idcs):
    """Permuted `n_idcs` while keepin.
    
    Paramaters
    ----------
    n_idcs : int or array-like of int
        Number of indices. If list, it should be a partion of the real number of idcs.
        Each partition will be permuted separately.
    """

    if isinstance(n_idcs, int):
        idcs = list(range(n_idcs))
    else:
        idcs = [list(range(partition)) for partition in n_idcs]

    if isinstance(n_idcs, int):
        random.shuffle(idcs)
        idcs = torch.tensor(idcs)

    else:

        # shuffle each partition separetly
        for partition_idcs in idcs:
            random.shuffle(partition_idcs)

        idcs = torch.cat([torch.tensor(idcs) for idcs in idcs])

    return idcs

class NaiveNuisanceGetter(nn.Module):
    """Class which return some random N in the y decomposition of X."""
    def __init__(self, n_heads, cardinality_X):
        nuisances = []
        for n_head in range(n_heads):
            idcs = list(range(cardinality_X))
            random.shuffle(idcs)
            nuisances.append(torch.tensor(idcs))
        self.register_buffer('nuisances', torch.stack(nuisances))

    def __call__(self, i, idcs):
        """Return the ith nuisance."""
        return self.nuisances[i,idcs]