"""Tests for Heston European call pricing."""
import numpy as np
import pytest
from heston.pricing import heston_call_price

PARAMS = (0.04, 1.5, 0.04, 0.5, -0.7)
S0, r, T = 100.0, 0.01, 1.0


def test_atm_price_is_positive():
    price = heston_call_price(S0, 100.0, T, r, PARAMS)
    assert price > 0.0


def test_atm_price_is_finite():
    price = heston_call_price(S0, 100.0, T, r, PARAMS)
    assert np.isfinite(price)


def test_call_above_intrinsic():
    """Call price must exceed the intrinsic value (no-arbitrage lower bound)."""
    K = 95.0
    price    = heston_call_price(S0, K, T, r, PARAMS)
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    assert price > intrinsic


def test_call_below_spot():
    """Call price cannot exceed the spot price."""
    price = heston_call_price(S0, 100.0, T, r, PARAMS)
    assert price < S0


def test_call_monotone_decreasing_in_strike():
    """Higher strike → lower call price."""
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices  = heston_call_price(S0, strikes, T, r, PARAMS)
    assert np.all(np.diff(prices) < 0.0)


def test_vectorised_matches_scalar():
    """Vectorised output must equal the scalar-loop result element-wise."""
    strikes = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
    prices_vec    = heston_call_price(S0, strikes, T, r, PARAMS)
    prices_scalar = np.array([heston_call_price(S0, K, T, r, PARAMS) for K in strikes])
    np.testing.assert_allclose(prices_vec, prices_scalar, rtol=1e-12)


def test_scalar_return_type():
    """Scalar K should return a Python/NumPy scalar, not an array."""
    price = heston_call_price(S0, 100.0, T, r, PARAMS)
    assert np.ndim(price) == 0


def test_array_return_type():
    """Array K should return an array of the same length."""
    strikes = np.linspace(80, 120, 5)
    prices  = heston_call_price(S0, strikes, T, r, PARAMS)
    assert prices.shape == (5,)


def test_short_maturity_still_finite():
    price = heston_call_price(S0, 100.0, 0.01, r, PARAMS)
    assert np.isfinite(price)
    assert price > 0.0
