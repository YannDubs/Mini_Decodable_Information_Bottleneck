from glob import glob
import os

import math
import torch.nn as nn
import random
import torch
from torch.distributions import Independent, Normal
import numpy as np

from functools import reduce
import operator


from torch.autograd import Function


def freeze_(module):
    """Freezes a pytorch module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def prod(iterable):
    """Take product of iterable like."""
    return reduce(operator.mul, iterable, 1)


def mean(l):
    """Take mean of array like."""
    return sum(l) / len(l)


class MLP(nn.Module):
    """Multi Layer Perceptron."""

    def __init__(self, dim_in, dim_out, n_hid_layers=2, dim_hid=128):
        super().__init__()
        # Leaky relu can help if deep
        layers = [nn.Linear(dim_in, dim_hid), nn.LeakyReLU()]
        for _ in range(n_hid_layers):
            layers += [nn.Linear(dim_hid, dim_hid), nn.LeakyReLU()]
        layers += [nn.Linear(dim_hid, dim_out)]
        self.module = nn.Sequential(*layers)
        self.initialize()

    def forward(self, x):
        return self.module(x)

    def initialize(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.kaiming_uniform_(m.weight, a=1e-2, nonlinearity="leaky_relu")


class ScaleGrad(Function):
    """Function which scales the gradients of the imputs by a scalar `lambd`."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(x)
        ctx.lambd = lambd
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            grad_output = grad_output * ctx.lambd
        return grad_output, None


scale_grad = ScaleGrad.apply


def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


def mean_logits(logits, dim=0):
    """Return the mean logit, where the average is taken over across `dim` in probability space."""
    # p = e^logit / (sum e^logit) <=> log(p) = logit - log_sum_exp(logit)
    log_prob = logits - torch.logsumexp(logits, -1, keepdim=True)
    # mean(p) = 1/Z sum exp(log(p)) <=> log(mean(p)) = log_sum_exp(log(p)) - log(Z)
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

    def __init__(self, n_heads, cardinality_X, cardinality_Y):
        super().__init__()
        nuisances = []
        for _ in range(n_heads):
            nuisances.append(
                torch.from_numpy(np.random.choice(cardinality_Y, cardinality_X))
            )
        self.register_buffer("nuisances", torch.stack(nuisances))

    def __call__(self, i, idcs):
        """Return the ith nuisance."""
        return self.nuisances[i, idcs]


class CrossEntropyLossGeneralize(nn.CrossEntropyLoss):
    """Cross entropy loss that forces (anti)-generalization.

    Note
    ----
    - we want to find an empirical risk minimizer that maximizes (antigeneralize) or minimizes
    (generalize) the test loss. Using a lagrangian relaxation of the problem this can be written
    as `min trainLoss + gamma * testLoss`, where the sign of `gamma` determines whether or not to
    generalize.
    - Each target should contain `(label, is_train)`. Where `is_train` says whether its a trainign
    example or a test example. `is_train=1` or `is_train=0`.

    Parameters
    ----------
    gamma : float, optional
        Langrangian coefficient of the relaxed problem. If positive, forces generalization, if negative
        forces anti generalization. Its scale balances the training and testing loss. If `gamma=0`
        becomes standard cross entropy.

    cap_test_loss : float, optional
        Value used to cap the test loss (i.e. don't backprop through it). This is especially useful
        when gamma is negative (anti generalization). Indeed, cross entropy is not bounded and thus
        the model could end up only focusing on maximizing the test loss to infinity regardless of
        train.

    kwargs :
        Additional arguments to `torch.nn.CrossEntropyLoss`.
    """

    def __init__(self, gamma=-0.1, cap_test_loss=10, **kwargs):
        super().__init__(reduction="none", **kwargs)
        self.gamma = gamma
        self.cap_test_loss = cap_test_loss

    def forward(self, inp, targets):
        label, is_train = targets
        out = super().forward(inp, label)

        if self.gamma == 0:
            # simple cross entropy
            return out.mean()

        is_test = is_train == 0
        weights = is_test.int() * self.gamma + is_train.int()

        # CAPPING : don't backprop if test and larger than cap (but still forward)
        is_large_loss = out > self.cap_test_loss
        to_cap = is_large_loss & is_test
        out[to_cap] = out[to_cap] * 0 + out[to_cap].detach()

        return (weights * out).mean()


def get_exponential_decay_gamma(scheduling_factor, max_epochs):
    """Return the exponential learning rate factor gamma.

    Parameters
    ----------
    scheduling_factor :
        By how much to reduce learning rate during training.

    max_epochs : int
        Maximum number of epochs.
    """
    return (1 / scheduling_factor) ** (1 / max_epochs)
