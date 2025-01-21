#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch_scatter import scatter_mean

from .transition import ContinuousTransition


def mean_flat(tensor, mask_generate, batch_ids):
    """
    Take the mean over all non-batch dimensions.
    """
    tensor, batch_ids = tensor[mask_generate], batch_ids[mask_generate] # [N, ...]
    tensor = torch.mean(tensor, dim=list(range(1, len(tensor.shape)))) # [N]
    return scatter_mean(tensor, batch_ids, dim=0) # [batch_size]


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales, delta=1.0e-2):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing with given delta

    :param x: the continuous data.
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + delta)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - delta)
    cdf_min = approx_standard_normal_cdf(min_in)
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.log(cdf_delta.clamp(min=1e-12))
    assert log_probs.shape == x.shape
    return log_probs


def _prior_bpd(self, x_start):
    """
    Get the prior KL term for the variational lower-bound, measured in
    bits-per-dim.

    This term can't be optimized, as it only depends on the encoder.

    :param x_start: the [N x C x ...] tensor of inputs.
    :return: a batch of [N] KL values (in bits), one per batch element.
    """
    batch_size = x_start.shape[0]
    t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
    qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
    kl_prior = normal_kl(
        mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
    )
    return mean_flat(kl_prior) / np.log(2.0)