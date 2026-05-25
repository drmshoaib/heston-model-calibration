"""Shared fixtures for the heston test suite."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def heston_params() -> tuple:
    """Standard Heston parameters used as ground truth throughout tests."""
    return (0.04, 1.5, 0.04, 0.5, -0.7)


@pytest.fixture
def market_setup() -> dict:
    return {"S0": 100.0, "r": 0.01}


@pytest.fixture
def simple_data(heston_params, market_setup):
    """Small synthetic dataset: 5 strikes at T = 1Y, ATM ± 10%."""
    from heston.implied_vol import implied_vol_call
    from heston.pricing import heston_call_price
    from scipy.stats import norm

    S0 = market_setup["S0"]
    r  = market_setup["r"]
    T  = 1.0
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]

    data = []
    for K in strikes:
        price = heston_call_price(S0, K, T, r, heston_params)
        iv    = implied_vol_call(S0, K, T, r, price)
        d1    = (np.log(S0 / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        vega  = S0 * norm.pdf(d1) * np.sqrt(T)
        data.append((K, T, float(iv), float(vega)))
    return data


@pytest.fixture
def default_bounds() -> np.ndarray:
    return np.array([
        [1e-4, 1.0],
        [0.01, 5.0],
        [1e-4, 1.0],
        [0.01, 2.0],
        [-0.99, 0.99],
    ])
