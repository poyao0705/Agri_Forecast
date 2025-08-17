"""
GARCH Data Generating Processes.

GARCH(1,1) models with different innovation distributions.
"""

import numpy as np


def simulate_garch11_t(
    n=6000, alpha=0.05, beta=0.94, nu=8, mu=0.0, sigma2_target=1e-4, burn=2000
):
    """
    Simulate GARCH(1,1) with t-distributed innovations.

    Parameters:
    -----------
    n : int
        Number of observations
    alpha : float
        ARCH parameter (coefficient on squared returns)
    beta : float
        GARCH parameter (coefficient on lagged variance)
    nu : float
        Degrees of freedom for t-distribution
    mu : float
        Mean return
    sigma2_target : float
        Target variance (unconditional)
    burn : int
        Burn-in period

    Returns:
    --------
    y : np.ndarray
        Simulated returns
    sigma_t : np.ndarray
        Conditional volatility
    """
    from scipy.stats import t

    omega = (1.0 - alpha - beta) * sigma2_target
    z = t.rvs(df=nu, size=n + burn) / np.sqrt(nu / (nu - 2))
    h = np.empty(n + burn)
    y = np.empty(n + burn)
    h[0] = sigma2_target

    for t_ in range(n + burn):
        y[t_] = mu + np.sqrt(max(h[t_], 1e-12)) * z[t_]
        if t_ + 1 < n + burn:
            h[t_ + 1] = omega + alpha * (y[t_] - mu) ** 2 + beta * h[t_]

    y = y[burn:]
    h = h[burn:]
    sigma_t = np.sqrt(h)
    return y, sigma_t


def simulate_garch11_skt(
    n=6000,
    alpha=0.05,  # GARCH alpha (ARCH)
    beta=0.94,  # GARCH beta
    nu=8,  # skew-t dof
    lam=-0.5,  # skewness in (-1, 1)
    mu=0.0,
    sigma2_target=1e-4,
    burn=2000,
):
    """
    Simulate GARCH(1,1) with skewed t-distributed innovations.

    Parameters:
    -----------
    n : int
        Number of observations
    alpha : float
        ARCH parameter (coefficient on squared returns)
    beta : float
        GARCH parameter (coefficient on lagged variance)
    nu : float
        Degrees of freedom for t-distribution
    lam : float
        Skewness parameter in (-1, 1)
    mu : float
        Mean return
    sigma2_target : float
        Target variance (unconditional)
    burn : int
        Burn-in period

    Returns:
    --------
    y : np.ndarray
        Simulated returns
    sigma_t : np.ndarray
        Conditional volatility
    """
    from scipy.stats import t

    omega = (1.0 - alpha - beta) * sigma2_target
    h = np.empty(n + burn)
    y = np.empty(n + burn)
    h[0] = sigma2_target

    # Skew-t innovations
    z_raw = t.rvs(df=nu, size=n + burn)
    z = z_raw / np.sqrt(nu / (nu - 2))  # normalize variance

    # Apply skewness transformation
    if abs(lam) > 1e-8:
        # Johnson's SU transformation for skewness
        delta = lam / np.sqrt(1 + lam**2)
        gamma = 1 / np.sqrt(1 + lam**2)
        z = delta + gamma * z
        # Re-normalize to unit variance
        z = (z - z.mean()) / z.std()

    for t_ in range(n + burn):
        y[t_] = mu + np.sqrt(max(h[t_], 1e-12)) * z[t_]
        if t_ + 1 < n + burn:
            h[t_ + 1] = omega + alpha * (y[t_] - mu) ** 2 + beta * h[t_]

    y = y[burn:]
    h = h[burn:]
    sigma_t = np.sqrt(h)
    return y, sigma_t
