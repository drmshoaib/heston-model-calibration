"""Convenience metrics helper (retained for backward compatibility).

For new code, prefer computing errors directly via
``heston_call_price`` + ``implied_vol_call`` and NumPy reductions.
"""
from __future__ import annotations

import numpy as np

from heston.implied_vol import implied_vol_call
from heston.pricing import heston_call_price


def iv_error_metrics(
    S0: float,
    r: float,
    params: tuple,
    market_data: list[tuple],
) -> tuple[float, float, float]:
    """Compute RMSE, MAE, and max absolute IV error.

    Parameters
    ----------
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    params : tuple
        Heston parameters ``(v0, kappa, theta, sigma, rho)``.
    market_data : list of tuple
        Each element is either ``(K, T, iv_mkt)`` or ``(K, T, iv_mkt, vega)``.

    Returns
    -------
    rmse, mae, max_error : float
        Errors in implied volatility units.
    """
    errors = []
    for row in market_data:
        K, T, iv_mkt = row[0], row[1], row[2]
        price  = heston_call_price(S0, K, T, r, params)
        iv_mod = implied_vol_call(S0, K, T, r, price)
        if not np.isnan(iv_mod):
            errors.append(iv_mod - iv_mkt)

    errors = np.array(errors, dtype=float)
    rmse   = float(np.sqrt(np.mean(errors**2)))
    mae    = float(np.mean(np.abs(errors)))
    maxe   = float(np.max(np.abs(errors)))
    return rmse, mae, maxe
