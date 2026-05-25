"""Heston pricing and calibration performance benchmark.

Compares scalar (per-strike loop) vs vectorised (batch) pricing throughput,
and measures single-start calibration runtime on a small synthetic surface.

Usage (from project root)::

    python examples/benchmark_pricing.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heston.calibration import calibrate_scipy
from heston.implied_vol import implied_vol_call
from heston.pricing import heston_call_price

PARAMS = (0.04, 1.5, 0.04, 0.5, -0.7)
S0, r, T = 100.0, 0.01, 1.0
N_REPEATS = 5


def benchmark_scalar(n_strikes=200):
    strikes = np.linspace(70.0, 130.0, n_strikes)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        for K in strikes:
            heston_call_price(S0, K, T, r, PARAMS)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def benchmark_vectorised(n_strikes=200):
    strikes = np.linspace(70.0, 130.0, n_strikes)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        heston_call_price(S0, strikes, T, r, PARAMS)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def benchmark_calibration():
    """Single-start calibration on 9 synthetic ATM ± 20% strikes."""
    strikes = np.linspace(80.0, 120.0, 9)
    data = []
    for K in strikes:
        price = heston_call_price(S0, K, T, r, PARAMS)
        iv    = implied_vol_call(S0, K, T, r, price)
        d1    = (np.log(S0 / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        vega  = float(S0 * norm.pdf(d1) * np.sqrt(T))
        data.append((float(K), float(T), float(iv), vega))

    bounds = np.array([
        [1e-4, 1.0],
        [0.01, 5.0],
        [1e-4, 1.0],
        [0.01, 2.0],
        [-0.99, 0.99],
    ])
    p0 = np.array([0.06, 1.0, 0.06, 0.4, -0.5])

    result = calibrate_scipy(p0, data, S0, r, bounds)
    return result["runtime"], result["nfev"], result["cost"], result["success"]


if __name__ == "__main__":
    N = 200
    sep = "=" * 60

    print(f"\n{sep}")
    print("Heston Pricing & Calibration Benchmark")
    print(f"{sep}")
    print(f"N_strikes = {N}   N_integration_nodes = 2000 (default)")
    print(f"Repeats   = {N_REPEATS}\n")

    mu_s, sd_s = benchmark_scalar(N)
    print(f"Scalar pricing   ({N} strikes, loop):       {mu_s*1e3:6.1f} ms  ± {sd_s*1e3:.1f} ms")

    mu_v, sd_v = benchmark_vectorised(N)
    print(f"Vectorised pricing ({N} strikes, batch):   {mu_v*1e3:6.1f} ms  ± {sd_v*1e3:.1f} ms")

    speedup = mu_s / mu_v if mu_v > 0 else float("nan")
    print(f"\nSpeedup (vectorised vs scalar):            {speedup:.1f}×")

    runtime, nfev, cost, converged = benchmark_calibration()
    print(f"\nSingle-start calibration (scipy TRF, 9 pts):")
    print(f"  Runtime     : {runtime * 1e3:.0f} ms")
    print(f"  Fevals      : {nfev}")
    print(f"  Final cost  : {cost:.2e}")
    print(f"  Converged   : {converged}")
    print(f"\n{sep}\n")
