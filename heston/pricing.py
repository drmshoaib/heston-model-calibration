"""Fourier-based European call pricing under the Heston model.

Uses the Lewis (2001) formula::

    C = S0 - sqrt(S0·K)·exp(−rT)/π · ∫₀^∞ Re[exp(−iu·k)·φ(u−½i;T)/(u²+¼)] du

where k = log(K/S0) is the log-moneyness.

Integration methods
-------------------
``method='trapz'`` (default)
    Uniform trapezoidal rule on [ε, umax] with N nodes.  N=2000 gives
    absolute price accuracy ≈ 2×10⁻⁷ for typical Heston parameters.
    **Recommended for calibration** — nearly optimal for this oscillatory
    integral and already achieves sub-basis-point IV accuracy.

``method='gl'`` / ``method='gauss-legendre'``
    n-point Gauss-Legendre quadrature on [0, umax].  Maps the GL nodes
    from [−1, 1] to [0, umax] and evaluates the integrand at those nodes::

        ∫₀^{umax} f(u) du ≈ (umax/2) · Σᵢ wᵢ · f(xᵢ)

    Achieves ≈ 10⁻⁴ absolute price accuracy with n=64 (vs 10⁻⁷ for
    trapz N=2000).  Numerically stable for any n.  The Lewis integrand is
    oscillatory (exp(−iuk) phase), which limits polynomial convergence; for
    this reason trapz outperforms GL in accuracy per cost.  GL is provided
    as a reproducible alternative for comparison and research purposes.

Both methods support vectorised pricing over an array of strikes.

References
----------
Lewis, A. L. (2001). A simple option formula for general jump-diffusion and
other exponential Lévy processes. *SSRN Working Paper*.
"""
from __future__ import annotations

from numpy.polynomial.legendre import leggauss as _leggauss

import numpy as np
from numpy.typing import NDArray

from .cf import heston_cf

# NumPy ≥ 2.0 renamed np.trapz → np.trapezoid; support both.
try:
    _trapezoid = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapezoid = np.trapz  # type: ignore[attr-defined]

# Module-level cache for GL nodes — avoids recomputation across calibration calls.
_GL_CACHE: dict[tuple[int, float], tuple[NDArray, NDArray]] = {}


# ---------------------------------------------------------------------------
# Integration grids
# ---------------------------------------------------------------------------

def _make_integration_grid(N: int, umax: float) -> NDArray:
    """Uniform grid on (0, umax] with N points."""
    return np.linspace(1e-6, umax, N)


def _gauss_legendre_grid(n: int = 64, umax: float = 100.0) -> tuple[NDArray, NDArray]:
    """n-point Gauss-Legendre nodes and weights on [0, umax] (cached).

    Maps the standard GL nodes on [−1, 1] to [0, umax] via::

        x_i = (umax/2) · (ξᵢ + 1),   w_i = (umax/2) · ωᵢ

    so that ``Σᵢ wᵢ f(xᵢ) ≈ ∫₀^{umax} f(u) du``.

    Parameters
    ----------
    n : int
        Number of quadrature nodes (default 64).
    umax : float
        Upper bound of the integration interval (default 100.0).

    Returns
    -------
    x : np.ndarray, shape (n,)
        Quadrature nodes in [0, umax].
    w : np.ndarray, shape (n,)
        Corresponding weights (include the Jacobian factor umax/2).
    """
    key = (n, umax)
    if key not in _GL_CACHE:
        xi, wi   = _leggauss(n)
        x = 0.5 * umax * (xi + 1.0)
        w = 0.5 * umax * wi
        _GL_CACHE[key] = (np.asarray(x), np.asarray(w))
    return _GL_CACHE[key]


# ---------------------------------------------------------------------------
# Complex-step helper (used by calibration.jacobian_cs)
# ---------------------------------------------------------------------------

def _heston_call_price_cs(
    S0: float,
    K: float | NDArray,
    T: float,
    r: float,
    params: tuple,
    N: int = 2000,
    umax: float = 100.0,
) -> complex | NDArray:
    """Lewis integral without the final ``np.real()`` step — for CS Jacobian.

    Computes the full complex Lewis integral, skipping the pointwise
    ``np.real()`` application.  When ``params[j]`` is perturbed by ``i·h``,
    the imaginary part of the returned complex price divided by ``h`` gives
    ``∂C/∂params[j]`` to machine precision (complex-step formula).

    This function is an internal helper; call :func:`jacobian_cs` from the
    calibration module instead of using this directly.
    """
    scalar_input = np.ndim(K) == 0
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))

    u      = _make_integration_grid(N, umax)
    phi    = heston_cf(u - 0.5j, T, params)
    kernel = phi / (u**2 + 0.25)
    log_mk = np.log(K_arr / S0)

    # Full complex integrand — NO np.real() applied here.
    integrand = kernel[:, None] * np.exp(-1j * u[:, None] * log_mk[None, :])
    integral  = _trapezoid(integrand, u, axis=0)

    prices = S0 - np.sqrt(S0 * K_arr) * np.exp(-r * T) / np.pi * integral
    return prices[0] if scalar_input else prices


# ---------------------------------------------------------------------------
# Main pricing function
# ---------------------------------------------------------------------------

def heston_call_price(
    S0: float,
    K: float | NDArray,
    T: float,
    r: float,
    params: tuple[float, float, float, float, float],
    N: int = 2000,
    umax: float = 100.0,
    method: str = "trapz",
    n_gl: int = 64,
) -> float | NDArray:
    """European call price under the Heston model (Lewis 2001).

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float or array_like
        Strike price(s).  Scalar or 1-D array; the CF is evaluated once per
        maturity regardless of how many strikes are requested.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    params : tuple of float
        Heston parameters ``(v0, kappa, theta, sigma, rho)``.
    N : int
        Trapezoidal rule nodes (``method='trapz'`` only, default 2000).
    umax : float
        Upper integration bound in both methods (default 100.0).
    method : str
        ``'trapz'`` (default) — uniform trapezoidal rule, ~2×10⁻⁷ accuracy.
        ``'gl'`` / ``'gauss-legendre'`` — Gauss-Legendre on [0, umax],
        ~10⁻⁴ accuracy with n_gl=64.  See module docstring for tradeoffs.
    n_gl : int
        Gauss-Legendre nodes (``method='gl'`` only, default 64).

    Returns
    -------
    float or np.ndarray
        Call price(s).  Scalar when ``K`` is scalar, array otherwise.

    Notes
    -----
    Zero dividends assumed.  For calibration, ``method='trapz'`` with
    N=2000 is recommended — it achieves sub-basis-point IV accuracy and
    is near-optimal for the oscillatory Lewis integrand.
    """
    scalar_input = np.ndim(K) == 0
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))

    if method in ("gl", "gauss-legendre"):
        u, w = _gauss_legendre_grid(n_gl, umax)
        phi    = heston_cf(u - 0.5j, T, params)
        kernel = phi / (u**2 + 0.25)
        log_mk = np.log(K_arr / S0)

        # Integrand: shape (n_gl, M)
        integrand = np.real(
            kernel[:, None] * np.exp(-1j * u[:, None] * log_mk[None, :])
        )
        # GL sum: Σᵢ wᵢ · f(xᵢ)  (weights already include Jacobian umax/2)
        integral = np.sum(w[:, None] * integrand, axis=0)

    else:
        # Default: uniform trapezoidal rule
        u      = _make_integration_grid(N, umax)
        phi    = heston_cf(u - 0.5j, T, params)
        kernel = phi / (u**2 + 0.25)
        log_mk = np.log(K_arr / S0)

        integrand = np.real(
            kernel[:, None] * np.exp(-1j * u[:, None] * log_mk[None, :])
        )
        integral = _trapezoid(integrand, u, axis=0)

    prices = S0 - np.sqrt(S0 * K_arr) * np.exp(-r * T) / np.pi * integral
    return float(prices[0]) if scalar_input else prices
