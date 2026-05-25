"""Heston model calibration via constrained nonlinear least squares.

Optimisation backends
---------------------
``calibrate_scipy``
    ``scipy.optimize.least_squares`` (TRF algorithm).  **Recommended default.**
    Supports multiple Jacobian options:
    - ``jac='3-point'``  — scipy central finite differences (default)
    - ``jac='analytic'`` — exact analytic Jacobian via chain rule through the CF;
                           eliminates all FD evaluations and achieves machine-
                           precision gradient accuracy

``levenberg_marquardt``
    Hand-rolled Levenberg–Marquardt loop.  Retained for reference and
    pedagogical comparison.

Multi-start
-----------
``multistart_calibrate``
    Sequential multi-start with reproducible random seeds.

``multistart_calibrate_parallel``
    Parallel multi-start using ``concurrent.futures.ProcessPoolExecutor``.
    Requires the caller to be guarded by ``if __name__ == '__main__':``
    on Windows (spawn semantics).

Jacobians
---------
``jacobian_analytic``
    Exact Jacobian computed by differentiating through the Riccati-ODE
    intermediates analytically.  Computes ``∂C/∂θⱼ`` via the Lewis integral
    with the CF gradient supplied by :func:`~heston.cf.heston_cf_and_grads`.
    No finite-difference evaluations.  CF evaluated once per unique maturity
    across all strikes.

``jacobian``
    Central finite-difference Jacobian (legacy, used by ``levenberg_marquardt``).

Objective
---------
Vega-weighted IV residuals::

    rᵢ(θ) = (IV_model(Kᵢ, Tᵢ; θ) − IV_mktᵢ) / vegaᵢ

Chain rule for Jacobian::

    ∂rᵢ/∂θⱼ = (1/vegaᵢ²) · ∂C_Heston(Kᵢ,Tᵢ;θ)/∂θⱼ
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .cf import heston_cf_and_grads
from .implied_vol import implied_vol_call
from .pricing import _heston_call_price_cs, _make_integration_grid, _trapezoid, heston_call_price
from .utils import enforce_bounds

logger = logging.getLogger(__name__)

# Default parameter bounds: (v0, kappa, theta, sigma, rho)
DEFAULT_BOUNDS: NDArray = np.array([
    [1e-4, 1.0],    # v0    — initial variance
    [0.01, 10.0],   # kappa — mean-reversion speed
    [1e-4, 1.0],    # theta — long-run variance
    [0.01, 2.0],    # sigma — vol of vol
    [-0.99, 0.99],  # rho   — spot–variance correlation
], dtype=float)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(
    params: NDArray,
    data: list[tuple[float, float, float, float]],
    S0: float,
    r: float,
) -> NDArray:
    """Vega-weighted IV residual vector.

    Parameters
    ----------
    params : np.ndarray
        ``(v0, kappa, theta, sigma, rho)``.
    data : list of (K, T, iv_mkt, vega)
        Market data tuples.
    S0 : float
        Spot price.
    r : float
        Risk-free rate.

    Returns
    -------
    np.ndarray, shape (m,)
        ``(IV_model − IV_mkt) / vega`` for each data point.
        Failed IV inversions (NaN) or zero/negative vega → penalty of 1.0.
    """
    res = []
    for K, T, iv_mkt, vega in data:
        price    = heston_call_price(S0, K, T, r, params)
        iv_model = implied_vol_call(S0, K, T, r, price)
        if np.isnan(iv_model) or vega <= 0.0:
            res.append(1.0)
        else:
            res.append((iv_model - iv_mkt) / vega)
    return np.array(res, dtype=float)


# ---------------------------------------------------------------------------
# Complex-step Jacobian
# ---------------------------------------------------------------------------

def jacobian_cs(
    params: NDArray,
    data: list[tuple[float, float, float, float]],
    S0: float,
    r: float,
    h: float = 1e-200,
) -> NDArray:
    """Complex-step Jacobian of the calibration residuals.

    .. warning::
        **This function is broken for the Lewis formula.**  The standard
        complex-step identity ``∂f/∂θ ≈ Im[f(θ+ih)]/h`` requires ``f`` to be
        real-valued when evaluated at real ``θ``.  The Lewis integral
        ``I(θ) = ∫ φ(u-½i;T)/(u²+¼)·exp(−iuk) du`` is complex-valued, so
        ``Im[I(θ+ih)]`` is O(1) rather than O(h), causing overflow
        (~10²⁰⁰) when divided by ``h = 1e-200``.  Use
        :func:`jacobian_analytic` instead.

    Parameters
    ----------
    params : np.ndarray, shape (5,)
        Current parameter vector.
    data : list of (K, T, iv_mkt, vega)
        Market data.
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    h : float
        Complex perturbation magnitude (default 1e-200).

    Returns
    -------
    np.ndarray, shape (m, 5)
        Jacobian matrix (incorrect values — retained for reference only).
    """
    n = len(params)
    m = len(data)
    J = np.zeros((m, n), dtype=float)

    for j in range(n):
        # Perturb params[j] with complex step
        p_cs = [complex(p) for p in params]
        p_cs[j] = complex(float(params[j]), h)

        for i, (K, T, _iv_mkt, vega) in enumerate(data):
            if vega <= 0.0:
                continue
            # C_cs is complex-valued; Im(C_cs)/h = ∂C/∂θⱼ
            price_cs = _heston_call_price_cs(S0, K, T, r, tuple(p_cs))
            dC_dtheta = np.imag(complex(price_cs)) / h
            # ∂rᵢ/∂θⱼ = (1/vegaᵢ²) · ∂Cᵢ/∂θⱼ
            J[i, j] = dC_dtheta / (vega * vega)

    return J


# ---------------------------------------------------------------------------
# Analytic Jacobian
# ---------------------------------------------------------------------------

def jacobian_analytic(
    params: NDArray,
    data: list[tuple[float, float, float, float]],
    S0: float,
    r: float,
    N: int = 2000,
    umax: float = 100.0,
) -> NDArray:
    """Exact Jacobian of calibration residuals via analytic CF differentiation.

    Computes ``∂rᵢ/∂θⱼ = (1/vegaᵢ²) · ∂Cᵢ/∂θⱼ`` where the price gradient::

        ∂C/∂θⱼ = −c · ∫ Re[(∂φ/∂θⱼ)/(u²+¼) · exp(−iuk)] du

    is evaluated with the analytic CF gradient from
    :func:`~heston.cf.heston_cf_and_grads`.  The characteristic function is
    evaluated **once per unique maturity** across all strikes, so the cost
    per Jacobian evaluation is O(unique_T × N) CF evaluations.

    Parameters
    ----------
    params : np.ndarray, shape (5,)
        ``(v0, kappa, theta, sigma, rho)``.
    data : list of (K, T, iv_mkt, vega)
        Market data tuples.
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    N : int
        Trapezoidal integration nodes (default 2000).
    umax : float
        Upper integration bound (default 100.0).

    Returns
    -------
    np.ndarray, shape (m, 5)
        Jacobian matrix with column order matching ``params``:
        (v0, kappa, theta, sigma, rho).
    """
    u     = _make_integration_grid(N, umax)   # shape (N,)
    denom = u**2 + 0.25                        # shape (N,)

    m = len(data)
    J = np.zeros((m, 5), dtype=float)

    # Group row indices by maturity so the CF is evaluated once per unique T
    T_to_indices: dict[float, list[int]] = {}
    for i, (_, T_val, _, _) in enumerate(data):
        T_to_indices.setdefault(T_val, []).append(i)

    for T_val, indices in T_to_indices.items():
        phi, dphi_dv0, dphi_dk, dphi_dt, dphi_ds, dphi_dr = heston_cf_and_grads(
            u - 0.5j, T_val, params
        )
        # Pre-divide each gradient by the denominator (u² + ¼)
        kernels = [g / denom for g in (dphi_dv0, dphi_dk, dphi_dt, dphi_ds, dphi_dr)]

        for i in indices:
            K, _, _, vega = data[i]
            if vega <= 0.0:
                continue
            c       = np.sqrt(S0 * K) * np.exp(-r * T_val) / np.pi
            log_mk  = np.log(K / S0)
            exp_iuk = np.exp(-1j * u * log_mk)   # shape (N,)

            for j, kernel in enumerate(kernels):
                integrand  = np.real(kernel * exp_iuk)
                J[i, j]    = -c * _trapezoid(integrand, u) / vega**2

    return J


# ---------------------------------------------------------------------------
# scipy.optimize.least_squares backend (recommended)
# ---------------------------------------------------------------------------

def calibrate_scipy(
    params0: NDArray | tuple,
    data: list[tuple[float, float, float, float]],
    S0: float,
    r: float,
    bounds: NDArray,
    method: str = "trf",
    max_nfev: int = 5000,
    gtol: float = 1e-8,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    jac: str = "3-point",
) -> dict[str, Any]:
    """Calibrate Heston parameters using ``scipy.optimize.least_squares``.

    Parameters
    ----------
    params0 : array_like
        Initial guess ``(v0, kappa, theta, sigma, rho)``.
    data : list of (K, T, iv_mkt, vega)
        Market data.
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    bounds : np.ndarray, shape (5, 2)
        Parameter bounds ``[[lo, hi], ...]``.
    method : str
        Optimisation algorithm: ``'trf'`` (default), ``'dogbox'``.
    max_nfev : int
        Maximum objective evaluations.
    gtol, ftol, xtol : float
        Convergence tolerances.
    jac : str
        Jacobian strategy:

        - ``'3-point'`` (default) — scipy central finite differences
        - ``'2-point'``           — scipy forward finite differences
        - ``'analytic'``          — exact analytic Jacobian via chain rule through
                                   the CF (:func:`jacobian_analytic`); machine-
                                   precision gradient, no FD evaluations

    Returns
    -------
    dict with keys
        ``params``    : calibrated parameter array (shape (5,))
        ``cost``      : 0.5 · Σ residuals²
        ``residuals`` : final residual vector
        ``nfev``      : number of objective evaluations
        ``success``   : bool
        ``message``   : solver status
        ``runtime``   : wall-clock time in seconds
        ``jac_method``: Jacobian strategy used
    """
    p0 = np.array(params0, dtype=float)
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    # Build Jacobian argument for least_squares
    if jac == "analytic":
        jac_arg = lambda p: jacobian_analytic(p, data, S0, r)
    elif jac == "cs":
        jac_arg = lambda p: jacobian_cs(p, data, S0, r)
    else:
        jac_arg = jac

    t0     = time.perf_counter()
    result = least_squares(
        lambda p: objective(p, data, S0, r),
        x0=p0,
        bounds=(lo, hi),
        method=method,
        jac=jac_arg,
        max_nfev=max_nfev,
        gtol=gtol,
        ftol=ftol,
        xtol=xtol,
    )
    elapsed = time.perf_counter() - t0

    logger.debug(
        "calibrate_scipy [jac=%s]: cost=%.4e  nfev=%d  converged=%s",
        jac, result.cost, result.nfev, result.success,
    )

    return {
        "params":     result.x,
        "cost":       float(result.cost),
        "residuals":  result.fun,
        "nfev":       result.nfev,
        "success":    result.success,
        "message":    result.message,
        "runtime":    elapsed,
        "jac_method": jac,
    }


# ---------------------------------------------------------------------------
# Sequential multi-start
# ---------------------------------------------------------------------------

def multistart_calibrate(
    data: list[tuple[float, float, float, float]],
    S0: float,
    r: float,
    bounds: NDArray,
    n_starts: int = 10,
    seed: int = 42,
    backend: str = "scipy",
    **backend_kwargs: Any,
) -> tuple[dict, list[dict]]:
    """Multi-start calibration with reproducible random initial guesses.

    Parameters
    ----------
    data : list of (K, T, iv_mkt, vega)
    S0, r : float
    bounds : np.ndarray, shape (5, 2)
    n_starts : int
        Number of random starting points (default 10).
    seed : int
        Random seed (default 42).
    backend : str
        ``'scipy'`` (recommended) or ``'lm'``.
    **backend_kwargs
        Forwarded to the backend.

    Returns
    -------
    best : dict
        Lowest-cost result.
    all_results : list of dict
        One dict per start.
    """
    rng   = np.random.default_rng(seed)
    lo    = bounds[:, 0]
    hi    = bounds[:, 1]
    n_dim = len(lo)

    all_results: list[dict] = []
    best: dict = {"cost": np.inf}

    for i in range(n_starts):
        p0 = lo + rng.random(n_dim) * (hi - lo)

        try:
            if backend == "scipy":
                res = calibrate_scipy(p0, data, S0, r, bounds, **backend_kwargs)
            else:
                t0     = time.perf_counter()
                params = levenberg_marquardt(p0, data, S0, r, bounds, **backend_kwargs)
                resid  = objective(params, data, S0, r)
                res    = {
                    "params":     params,
                    "cost":       float(0.5 * np.sum(resid**2)),
                    "residuals":  resid,
                    "nfev":       None,
                    "success":    True,
                    "message":    "LM finished",
                    "runtime":    time.perf_counter() - t0,
                    "jac_method": "fd",
                }
        except Exception as exc:
            logger.warning("Start %d failed: %s", i + 1, exc)
            res = {
                "params":     None,
                "cost":       np.inf,
                "residuals":  None,
                "nfev":       None,
                "success":    False,
                "message":    str(exc),
                "runtime":    0.0,
                "jac_method": "---",
            }

        res["start"] = i + 1
        all_results.append(res)

        if res["cost"] < best["cost"]:
            best = res

        logger.info(
            "Start %d/%d — cost=%.4e  converged=%s",
            i + 1, n_starts, res["cost"], res["success"],
        )

    return best, all_results


# ---------------------------------------------------------------------------
# Module-level worker (must be at module scope for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _calibration_worker(
    args: tuple,
) -> dict:
    """Single calibration run — module-level for ProcessPoolExecutor pickling."""
    i, p0, data, S0, r, bounds, kwargs = args
    try:
        res = calibrate_scipy(p0, data, S0, r, bounds, **kwargs)
    except Exception as exc:
        res = {
            "params":     None,
            "cost":       np.inf,
            "residuals":  None,
            "nfev":       None,
            "success":    False,
            "message":    str(exc),
            "runtime":    0.0,
            "jac_method": "---",
        }
    res["start"] = i + 1
    return res


# ---------------------------------------------------------------------------
# Parallel multi-start
# ---------------------------------------------------------------------------

def multistart_calibrate_parallel(
    data: list[tuple[float, float, float, float]],
    S0: float,
    r: float,
    bounds: NDArray,
    n_starts: int = 10,
    seed: int = 42,
    max_workers: int | None = None,
    **calibrate_kwargs: Any,
) -> tuple[dict, list[dict]]:
    """Parallel multi-start calibration via ``ProcessPoolExecutor``.

    Distributes calibration starts across CPU cores.  Each worker runs
    :func:`calibrate_scipy` independently and returns its result; the
    best result (lowest cost) is selected.

    .. note::
        On **Windows**, scripts that call this function must be guarded by
        ``if __name__ == '__main__':`` to prevent recursive spawning.
        Library code (non-script imports) does not require this guard.

    Parameters
    ----------
    data : list of (K, T, iv_mkt, vega)
    S0, r : float
    bounds : np.ndarray, shape (5, 2)
    n_starts : int
        Number of calibration starts (default 10).
    seed : int
        Random seed for reproducible starting points (default 42).
    max_workers : int or None
        Number of parallel workers.  ``None`` → use all available CPUs.
    **calibrate_kwargs
        Forwarded to :func:`calibrate_scipy` for each start.

    Returns
    -------
    best : dict
        Lowest-cost result dict (same structure as :func:`calibrate_scipy`).
    all_results : list of dict
        One result dict per start, ordered by start index.
    """
    rng   = np.random.default_rng(seed)
    lo    = bounds[:, 0]
    hi    = bounds[:, 1]
    n_dim = len(lo)

    starting_points = [lo + rng.random(n_dim) * (hi - lo) for _ in range(n_starts)]

    all_args = [
        (i, p0, data, S0, r, bounds, calibrate_kwargs)
        for i, p0 in enumerate(starting_points)
    ]

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_calibration_worker, all_args))
    except Exception as exc:
        logger.warning(
            "ProcessPoolExecutor failed (%s); falling back to sequential.", exc
        )
        results = [_calibration_worker(args) for args in all_args]

    best = min(results, key=lambda res: res["cost"])
    return best, results


# ---------------------------------------------------------------------------
# Finite-difference Jacobian (used by Levenberg-Marquardt)
# ---------------------------------------------------------------------------

def jacobian(
    params: NDArray,
    data: list[tuple[float, float, float, float]],
    S0: float,
    r: float,
    h: float = 1e-4,
) -> NDArray:
    """Central-difference Jacobian of ``objective`` w.r.t. ``params``."""
    n = len(params)
    m = len(data)
    J = np.zeros((m, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        J[:, j] = (
            objective(params + h * e, data, S0, r)
            - objective(params - h * e, data, S0, r)
        ) / (2.0 * h)
    return J


# ---------------------------------------------------------------------------
# Levenberg-Marquardt (hand-rolled, retained for reference)
# ---------------------------------------------------------------------------

def levenberg_marquardt(
    params0: NDArray | tuple,
    data: list[tuple[float, float, float, float]],
    S0: float,
    r: float,
    bounds: NDArray,
    max_iter: int = 50,
    lambda0: float = 1e-3,
    tol: float = 1e-6,
) -> NDArray:
    """Custom Levenberg–Marquardt calibration (hand-rolled).

    Retained for pedagogical reference.  For production use, prefer
    :func:`calibrate_scipy`.

    Parameters
    ----------
    params0 : array_like
        Initial guess.
    data : list of (K, T, iv_mkt, vega)
    S0, r : float
    bounds : np.ndarray, shape (5, 2)
    max_iter : int
        Maximum iterations (default 50).
    lambda0 : float
        Initial damping (default 1e-3).
    tol : float
        Step-norm convergence tolerance (default 1e-6).

    Returns
    -------
    np.ndarray, shape (5,)
        Calibrated parameters.
    """
    params = np.array(params0, dtype=float)
    lam    = lambda0

    for iteration in range(max_iter):
        res  = objective(params, data, S0, r)
        J    = jacobian(params, data, S0, r)
        A    = J.T @ J + lam * np.eye(len(params))
        g    = J.T @ res
        step = np.linalg.solve(A, g)

        new_params = enforce_bounds(params - step, bounds)

        if np.linalg.norm(step) < tol:
            logger.debug("LM converged at iteration %d (||step||=%.2e)",
                         iteration, np.linalg.norm(step))
            break

        new_cost = np.sum(objective(new_params, data, S0, r)**2)
        if np.sum(res**2) > new_cost:
            params = new_params
            lam   *= 0.5
        else:
            lam   *= 2.0

    return params
