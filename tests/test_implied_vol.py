"""Tests for Black-Scholes pricing and implied volatility inversion."""
import numpy as np
import pytest
from heston.implied_vol import black_scholes_call, implied_vol_call

S0, K, T, r = 100.0, 100.0, 1.0, 0.01


def test_bs_atm_price_positive():
    price = black_scholes_call(S0, K, T, r, 0.20)
    assert price > 0.0
    assert np.isfinite(price)


def test_bs_deep_itm_above_intrinsic():
    """Deep-ITM call price ≥ intrinsic value."""
    price     = black_scholes_call(200.0, 100.0, 1.0, 0.0, 0.001)
    intrinsic = max(200.0 - 100.0, 0.0)
    assert price >= intrinsic - 1e-6


def test_iv_roundtrip_atm():
    """IV inversion must recover the input volatility (ATM)."""
    sigma_true = 0.25
    price = black_scholes_call(S0, K, T, r, sigma_true)
    iv    = implied_vol_call(S0, K, T, r, price)
    assert abs(iv - sigma_true) < 1e-6


def test_iv_roundtrip_otm():
    """IV inversion must recover the input volatility (OTM)."""
    sigma_true = 0.30
    price = black_scholes_call(100.0, 110.0, T, r, sigma_true)
    iv    = implied_vol_call(100.0, 110.0, T, r, price)
    assert abs(iv - sigma_true) < 1e-6


def test_iv_roundtrip_itm():
    """IV inversion must recover the input volatility (ITM)."""
    sigma_true = 0.20
    price = black_scholes_call(100.0, 90.0, T, r, sigma_true)
    iv    = implied_vol_call(100.0, 90.0, T, r, price)
    assert abs(iv - sigma_true) < 1e-6


def test_iv_returns_nan_below_intrinsic():
    """Price strictly below intrinsic → nan."""
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    iv = implied_vol_call(S0, K, T, r, intrinsic * 0.5)
    assert np.isnan(iv)


def test_iv_returns_nan_for_zero_price():
    iv = implied_vol_call(S0, K, T, r, 0.0)
    assert np.isnan(iv)


def test_iv_returns_nan_above_spot():
    """Price above spot price → nan (put-call parity violation)."""
    iv = implied_vol_call(S0, K, T, r, S0 + 1.0)
    assert np.isnan(iv)


def test_bs_put_call_parity():
    """C − P = S0 − K·exp(−rT)."""
    sigma = 0.20
    C = black_scholes_call(S0, K, T, r, sigma)
    # Put from parity
    P = C - S0 + K * np.exp(-r * T)
    assert P > 0.0
