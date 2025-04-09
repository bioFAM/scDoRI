#################################
# utils.py
#################################
import torch
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

def set_seed(seed=200):
    """
    Set the random seed for Python, NumPy, and PyTorch (including CUDA if available).

    Parameters
    ----------
    seed : int, optional
        The desired random seed. Default is 200.

    Returns
    -------
    None
        Modifies global states of Python, NumPy, and PyTorch seeds in place.

    Notes
    -----
    Useful for ensuring reproducible results across runs when training or testing
    the model. However, full reproducibility can still be subject to GPU hardware
    determinism settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}.")

def log_nb_positive(
    x,
    mu,
    theta,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
):
    """
    Compute the log-likelihood for a Negative Binomial (NB) distribution.

    This function is often used for modeling overdispersed count data in
    scRNA-seq .

    Parameters
    ----------
    x : torch.Tensor
        Observed count data, shape (batch_size, num_features).
    mu : torch.Tensor
        Mean of the negative binomial, must be > 0. Same shape as x.
    theta : torch.Tensor
        Inverse-dispersion (overdispersion) parameter, must be > 0. Same shape as x.
    eps : float, optional
        A small constant for numerical stability in logarithms. Default is 1e-8.
    log_fn : callable, optional
        A function to take the logarithm, typically `torch.log`. Default is `torch.log`.
    lgamma_fn : callable, optional
        A function for computing log-gamma, typically `torch.lgamma`. Default is `torch.lgamma`.

    Returns
    -------
    torch.Tensor
        Element-wise log-likelihood of shape (batch_size, num_features).

    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res
