"""
Gaussian Data Generating Processes.

Simple IID Gaussian processes for baseline testing.
"""

import numpy as np


def simulate_iid_gaussian(n=6000, mu=0.0, sigma=0.01):
    """
    Simulate IID Gaussian returns.

    Parameters:
    -----------
    n : int
        Number of observations
    mu : float
        Mean return
    sigma : float
        Standard deviation of returns

    Returns:
    --------
    y : np.ndarray
        Simulated returns
    sigma_t : np.ndarray
        Constant volatility (sigma for all t)
    """
    eps = np.random.randn(n)
    y = mu + sigma * eps
    sigma_t = np.full(n, sigma)
    return y, sigma_t
