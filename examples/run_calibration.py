"""Minimal reproducible Heston calibration example.

Generates a noiseless synthetic smile from known parameters, calibrates
using the scipy.optimize.least_squares backend, and prints a parameter
comparison table and IV fit metrics.

Usage (from project root)::

    python examples/run_calibration.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heston.calibration import calibrate_scipy
from heston.implied_vol import implied_vol_call
from heston.pricing import heston_call_price

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Experiment configuration ─────────────────────────────────────────────────

TRUE_PARAMS = (0.04, 1.5, 0.04, 0.5, -0.7)   # (v0, kappa, theta, sigma, rho)
S0, r       = 100.0, 0.01
STRIKES     = np.linspace(80.0, 120.0, 9)
T           = 1.0

INITIAL_GUESS = np.array([0.06, 1.0, 0.06, 0.4, -0.5])

BOUNDS = np.array([
    [1e-4, 1.0],
    [0.01, 5.0],
    [1e-4, 1.0],
    [0.01, 2.0],
    [-0.99, 0.99],
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_data(params, strikes, T):
    data = []
    for K in strikes:
        price = heston_call_price(S0, K, T, r, params)
        iv    = implied_vol_call(S0, K, T, r, price)
        d1    = (np.log(S0 / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        vega  = S0 * norm.pdf(d1) * np.sqrt(T)
        data.append((float(K), float(T), float(iv), float(vega)))
    return data


def iv_rmse(params, data):
    errs = []
    for K, T_i, iv_mkt, _ in data:
        price  = heston_call_price(S0, K, T_i, r, params)
        iv_mod = implied_vol_call(S0, K, T_i, r, price)
        if not np.isnan(iv_mod):
            errs.append(iv_mod - iv_mkt)
    return float(np.sqrt(np.mean(np.array(errs) ** 2)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info(
        "Generating synthetic market data (%d strikes, T=%.1fY)...",
        len(STRIKES), T,
    )
    data = make_data(TRUE_PARAMS, STRIKES, T)

    log.info("Calibrating with scipy.optimize.least_squares (TRF)...")
    result = calibrate_scipy(INITIAL_GUESS, data, S0, r, BOUNDS)
    params = result["params"]
    rmse   = iv_rmse(params, data)

    names = ["v0", "kappa", "theta", "sigma", "rho"]
    print()
    print("=" * 52)
    print(f"{'Parameter':<12} {'True':>10} {'Calibrated':>12}")
    print("-" * 52)
    for name, true, cal in zip(names, TRUE_PARAMS, params):
        print(f"{name:<12} {true:>10.4f} {cal:>12.4f}")
    print("=" * 52)
    print(f"\nIV RMSE       : {rmse:.2e}")
    print(f"Function evals: {result['nfev']}")
    print(f"Runtime       : {result['runtime'] * 1000:.0f} ms")
    print(f"Converged     : {result['success']}")
    print()

    if not result["success"]:
        log.warning("Solver did not fully converge: %s", result["message"])


if __name__ == "__main__":
    main()
