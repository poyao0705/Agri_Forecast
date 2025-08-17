"""
Data Generating Processes (DGPs) for model testing and validation.

This package contains various simulated data generators for testing
VaR/ES prediction models under controlled conditions.
"""

from .gaussian import simulate_iid_gaussian
from .garch import simulate_garch11_t, simulate_garch11_skt
from .stochastic_volatility import simulate_sv
from .neural_like import simulate_srnn_like

# DGP registry for easy access
DGPS = {
    "iid_gaussian": simulate_iid_gaussian,
    "garch11_t": simulate_garch11_t,
    "garch11_skt": simulate_garch11_skt,
    "sv": simulate_sv,
    "srnn_like": simulate_srnn_like,
}

# Skew-t presets for garch11_skt
SKT_PRESETS = {
    "heavy": (5, -0.5),  # heavy tail, negative skew
    "light": (15, 0.2),  # light tail, positive skew
    "symmetric": (8, 0.0),  # symmetric
}

__all__ = [
    "DGPS",
    "SKT_PRESETS",
    "simulate_iid_gaussian",
    "simulate_garch11_t",
    "simulate_garch11_skt",
    "simulate_sv",
    "simulate_srnn_like",
]
