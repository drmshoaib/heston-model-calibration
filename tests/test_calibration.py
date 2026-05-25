"""Tests for calibration objective, scipy backend, LM, and multi-start."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from heston.calibration import (
    DEFAULT_BOUNDS,
    calibrate_scipy,
    jacobian_analytic,
    levenberg_marquardt,
    multistart_calibrate,
    objective,
)
from heston.implied_vol import implied_vol_call
from heston.pricing import heston_call_price

TRUE_PARAMS = (0.04, 1.5, 0.04, 0.5, -0.7)
S0, r = 100.0, 0.01

BOUNDS = np.array([
    [1e-4, 1.0],
    [0.01, 5.0],
    [1e-4, 1.0],
    [0.01, 2.0],
    [-0.99, 0.99],
])


def _make_data(params=TRUE_PARAMS, strikes=None, T=1.0):
    if strikes is None:
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    data = []
    for K in strikes:
        price = heston_call_price(S0, K, T, r, params)
        iv    = implied_vol_call(S0, K, T, r, price)
        d1    = (np.log(S0 / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        vega  = S0 * norm.pdf(d1) * np.sqrt(T)
        data.append((float(K), float(T), float(iv), float(vega)))
    return data


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def test_objective_returns_finite_array():
    data = _make_data()
    res  = objective(np.array(TRUE_PARAMS), data, S0, r)
    assert res.shape == (len(data),)
    assert np.all(np.isfinite(res))


def test_objective_near_zero_at_ground_truth():
    """Residuals should be essentially zero at the true parameters."""
    data = _make_data()
    res  = objective(np.array(TRUE_PARAMS), data, S0, r)
    assert np.max(np.abs(res)) < 1e-4


def test_objective_handles_nan_iv_gracefully():
    """Bad pricing should give penalty=1.0, not raise."""
    # Extremely OTM strike almost guaranteed to fail IV inversion
    bad_data = [(300.0, 1.0, 0.30, 0.01)]
    res = objective(np.array(TRUE_PARAMS), bad_data, S0, r)
    assert np.isfinite(res[0])


# ---------------------------------------------------------------------------
# scipy backend
# ---------------------------------------------------------------------------

def test_calibrate_scipy_returns_dict_keys():
    data   = _make_data()
    p0     = np.array([0.06, 1.0, 0.06, 0.4, -0.5])
    result = calibrate_scipy(p0, data, S0, r, BOUNDS)
    for key in ("params", "cost", "residuals", "nfev", "success", "runtime"):
        assert key in result


def test_calibrate_scipy_params_in_bounds():
    data   = _make_data()
    p0     = np.array([0.06, 1.0, 0.06, 0.4, -0.5])
    result = calibrate_scipy(p0, data, S0, r, BOUNDS)
    params = result["params"]
    assert np.all(params >= BOUNDS[:, 0] - 1e-10)
    assert np.all(params <= BOUNDS[:, 1] + 1e-10)


def test_calibrate_scipy_low_surface_rmse():
    """On noise-free synthetic data the surface RMSE should be < 1 vol point."""
    strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    data    = _make_data(strikes=strikes)
    p0      = np.array([0.06, 1.0, 0.06, 0.4, -0.5])
    result  = calibrate_scipy(p0, data, S0, r, BOUNDS)
    params  = result["params"]

    errs = []
    for K, T, iv_mkt, _ in data:
        price  = heston_call_price(S0, K, T, r, params)
        iv_mod = implied_vol_call(S0, K, T, r, price)
        if not np.isnan(iv_mod):
            errs.append(iv_mod - iv_mkt)

    rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))
    assert rmse < 0.01, f"RMSE = {rmse:.4f} exceeds 1 vol point"


# ---------------------------------------------------------------------------
# Levenberg–Marquardt (reference backend)
# ---------------------------------------------------------------------------

def test_lm_returns_ndarray_shape():
    data   = _make_data()
    p0     = np.array([0.06, 1.0, 0.06, 0.4, -0.5])
    params = levenberg_marquardt(p0, data, S0, r, BOUNDS)
    assert isinstance(params, np.ndarray)
    assert params.shape == (5,)


def test_lm_params_in_bounds():
    data   = _make_data()
    p0     = np.array([0.06, 1.0, 0.06, 0.4, -0.5])
    params = levenberg_marquardt(p0, data, S0, r, BOUNDS)
    assert np.all(params >= BOUNDS[:, 0] - 1e-10)
    assert np.all(params <= BOUNDS[:, 1] + 1e-10)


# ---------------------------------------------------------------------------
# Multi-start
# ---------------------------------------------------------------------------

def test_multistart_returns_correct_number_of_results():
    data = _make_data()
    _, all_results = multistart_calibrate(data, S0, r, BOUNDS, n_starts=3, seed=42)
    assert len(all_results) == 3


def test_multistart_best_has_minimum_cost():
    data = _make_data()
    best, all_results = multistart_calibrate(data, S0, r, BOUNDS, n_starts=4, seed=0)
    finite_costs = [res["cost"] for res in all_results if np.isfinite(res["cost"])]
    if finite_costs:
        assert best["cost"] <= min(finite_costs) + 1e-12


def test_multistart_reproducible():
    """Same seed must produce identical results across two calls."""
    data = _make_data()
    best1, _ = multistart_calibrate(data, S0, r, BOUNDS, n_starts=2, seed=99)
    best2, _ = multistart_calibrate(data, S0, r, BOUNDS, n_starts=2, seed=99)
    np.testing.assert_array_equal(best1["params"], best2["params"])


# ---------------------------------------------------------------------------
# Analytic Jacobian
# ---------------------------------------------------------------------------

def test_jacobian_analytic_shape():
    data = _make_data()
    J = jacobian_analytic(np.array(TRUE_PARAMS), data, S0, r)
    assert J.shape == (len(data), 5)
    assert np.all(np.isfinite(J))


def test_jacobian_analytic_matches_fd():
    """Analytic Jacobian must agree with central-FD Jacobian to within 0.5%."""
    data   = _make_data(strikes=[90.0, 95.0, 100.0, 105.0, 110.0])
    params = np.array(TRUE_PARAMS)

    J_an = jacobian_analytic(params, data, S0, r)

    # Central-FD Jacobian of the residual objective
    h   = 1e-5
    m, n = len(data), 5
    J_fd = np.zeros((m, n))
    for j in range(n):
        e         = np.zeros(n)
        e[j]      = h
        J_fd[:, j] = (objective(params + e, data, S0, r)
                      - objective(params - e, data, S0, r)) / (2.0 * h)

    # Relative error (normalised by max(|J_fd|, 1e-10) to handle near-zero entries)
    scale   = np.maximum(np.abs(J_fd), 1e-10)
    rel_err = np.abs(J_an - J_fd) / scale
    assert np.max(rel_err) < 0.005, (
        f"Max relative error between analytic and FD Jacobians = {np.max(rel_err):.2e}"
    )


def test_calibrate_scipy_analytic_jac_converges():
    """calibrate_scipy with jac='analytic' should reach low cost on synthetic data."""
    strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
    data    = _make_data(strikes=strikes)
    p0      = np.array([0.06, 1.0, 0.06, 0.4, -0.5])
    result  = calibrate_scipy(p0, data, S0, r, BOUNDS, jac="analytic")
    assert result["success"], f"Optimiser did not converge: {result['message']}"
    assert result["cost"] < 1e-8, f"Final cost {result['cost']:.2e} exceeds threshold"
