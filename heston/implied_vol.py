"""Black-Scholes pricing and implied volatility inversion.

Assumes European call options with zero dividends throughout.
Implied volatility is inverted using Brent's method (scipy.optimize.brentq),
which is more reliable than bisection for near-boundary cases.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black_scholes_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Black-Scholes European call price (no dividends).

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Implied / realised volatility (annualised, > 0).

    Returns
    -------
    float
        Call option price.
    """
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def implied_vol_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    price: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Implied volatility of a European call via Brent's method.

    Inverts the Black-Scholes formula to recover the volatility σ such that
    ``black_scholes_call(S0, K, T, r, σ) == price``.

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    price : float
        Observed call price (from market or from a pricing model).
    tol : float
        Absolute tolerance on the recovered volatility (default 1e-8).
    max_iter : int
        Maximum number of function evaluations passed to brentq (default 100).

    Returns
    -------
    float
        Implied volatility, or ``np.nan`` if inversion fails.

    Notes
    -----
    Returns ``np.nan`` when ``price`` violates no-arbitrage bounds, i.e.
    ``price <= intrinsic_value`` or ``price >= S0``.  This prevents
    calibration from crashing on a single bad pricing evaluation.

    The search interval for σ is [1e-8, 5.0].  Volatilities outside this
    range are unlikely in practice; returning nan is preferable to silently
    extrapolating.
    """
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)

    # No-arbitrage bounds check
    if price <= intrinsic or price >= S0:
        return np.nan

    sigma_lo, sigma_hi = 1e-8, 5.0

    # Verify the search interval brackets a root
    f_lo = black_scholes_call(S0, K, T, r, sigma_lo) - price
    f_hi = black_scholes_call(S0, K, T, r, sigma_hi) - price
    if f_lo >= 0 or f_hi <= 0:
        return np.nan

    try:
        iv = brentq(
            lambda sigma: black_scholes_call(S0, K, T, r, sigma) - price,
            sigma_lo,
            sigma_hi,
            xtol=tol,
            maxiter=max_iter,
            full_output=False,
        )
        return float(iv)
    except ValueError:
        return np.nan
