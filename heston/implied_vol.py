import numpy as np
from scipy.special import erf

def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N = lambda x: 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    return S0 * N(d1) - K * np.exp(-r * T) * N(d2)


def implied_vol_call(
    S0,
    K,
    T,
    r,
    price,
    tol=1e-6,
    max_iter=100
):
    """
    Implied volatility via bisection.
    Returns np.nan if inversion fails.
    """

    # No-arbitrage bounds
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    upper_bound = S0

    if price < intrinsic or price > upper_bound:
        return np.nan

    low, high = 1e-8, 5.0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        bs_price = black_scholes_call(S0, K, T, r, mid)

        if abs(bs_price - price) < tol:
            return mid

        if bs_price > price:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)
