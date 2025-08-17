"""
Neural Network-like Data Generating Processes.

Nonlinear process that mimics neural network behavior.
"""

import numpy as np


def simulate_srnn_like(n=6000, ah=0.98, ax=0.2, w=0.5, b=-3.0):
    """
    Simulate a neural network-like process.

    The model is:
    h_t = ah * h_{t-1} + ax * y_{t-1}^2
    sigma_t = log(1 + exp(w * h_t + b))  # softplus activation
    y_t = sigma_t * epsilon_t

    where epsilon_t ~ N(0,1)

    Parameters:
    -----------
    n : int
        Number of observations
    ah : float
        Hidden state persistence
    ax : float
        Coefficient on squared returns
    w : float
        Weight in softplus function
    b : float
        Bias in softplus function

    Returns:
    --------
    y : np.ndarray
        Simulated returns
    sigma_t : np.ndarray
        Conditional volatility
    """
    h_state = np.zeros(n)
    y = np.zeros(n)
    sigma_t = np.zeros(n)

    for t_ in range(1, n):
        h_state[t_] = ah * h_state[t_ - 1] + ax * y[t_ - 1] ** 2
        sigma_t[t_] = np.log(1 + np.exp(w * h_state[t_] + b))  # softplus
        y[t_] = sigma_t[t_] * np.random.randn()

    sigma_t[0] = sigma_t[1]  # Set first value
    return y, sigma_t
