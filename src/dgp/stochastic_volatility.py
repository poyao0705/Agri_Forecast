"""
Stochastic Volatility Data Generating Processes.

Log-normal stochastic volatility model.
"""

import numpy as np


def simulate_sv(n=6000, mu=-1.0, phi=0.98, sigma_eta=0.2):
    """
    Simulate stochastic volatility model.

    The model is:
    y_t = sigma_t * epsilon_t
    log(sigma_t) = mu + phi * log(sigma_{t-1}) + eta_t

    where epsilon_t ~ N(0,1) and eta_t ~ N(0, sigma_eta^2)

    Parameters:
    -----------
    n : int
        Number of observations
    mu : float
        Mean of log volatility
    phi : float
        Persistence parameter (should be close to 1)
    sigma_eta : float
        Standard deviation of volatility innovations

    Returns:
    --------
    y : np.ndarray
        Simulated returns
    sigma_t : np.ndarray
        Conditional volatility
    """
    log_sig = np.zeros(n)
    log_sig[0] = mu
    eta = np.random.randn(n) * sigma_eta

    for t_ in range(1, n):
        log_sig[t_] = mu + phi * (log_sig[t_ - 1] - mu) + eta[t_]

    sigma_t = np.exp(log_sig)
    y = sigma_t * np.random.randn(n)
    return y, sigma_t
