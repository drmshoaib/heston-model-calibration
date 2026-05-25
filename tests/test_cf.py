"""Tests for the Heston characteristic function."""
import numpy as np
import pytest
from heston.cf import heston_cf

PARAMS = (0.04, 1.5, 0.04, 0.5, -0.7)


def test_cf_scalar_real_input_is_finite():
    phi = heston_cf(1.0, 1.0, PARAMS)
    assert np.isfinite(np.real(phi))
    assert np.isfinite(np.imag(phi))


def test_cf_array_input_all_finite():
    u = np.linspace(0.1, 20.0, 100)
    phi = heston_cf(u, 1.0, PARAMS)
    assert phi.shape == (100,)
    assert np.all(np.isfinite(np.abs(phi)))


def test_cf_at_zero_is_one():
    """φ(0; T) = E[e^0] = 1 for any T."""
    phi = heston_cf(0.0, 1.0, PARAMS)
    assert abs(phi - 1.0) < 1e-10


def test_cf_complex_shift_all_finite():
    """Lewis formula uses u − 0.5j; the CF must be finite there."""
    u = np.linspace(1e-4, 100.0, 2000) - 0.5j
    phi = heston_cf(u, 1.0, PARAMS)
    assert np.all(np.isfinite(np.abs(phi)))


def test_cf_multiple_maturities():
    for T in [0.1, 0.25, 0.5, 1.0, 2.0]:
        phi = heston_cf(1.0, T, PARAMS)
        assert np.isfinite(np.real(phi)), f"CF non-finite for T={T}"


def test_cf_modulus_bounded_by_one_at_real_u():
    """For purely real u, |φ(u)| ≤ 1 (characteristic function property)."""
    u = np.linspace(0.1, 50.0, 200)
    phi = heston_cf(u, 1.0, PARAMS)
    assert np.all(np.abs(phi) <= 1.0 + 1e-10)
