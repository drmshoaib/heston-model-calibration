"""Heston characteristic function.

Branch convention: Gatheral (2006) / "Little Trap" formulation.
``Re(d) > 0`` is enforced to avoid branch-cut discontinuities that arise
from the complex square root when the argument crosses the negative real axis
during integration.  See Albrecher et al. (2007) for a detailed discussion.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def heston_cf(
    u: complex | NDArray,
    T: float,
    params: tuple[float, float, float, float, float],
) -> complex | NDArray:
    """Characteristic function of log(S_T) under the Heston model.

    Computes φ(u; T) = E[exp(iu · log S_T)] under the risk-neutral measure,
    using the affine (Riccati) representation of Heston (1993).

    Parameters
    ----------
    u : complex or array_like
        Frequency argument. May be complex-valued (e.g. ``u - 0.5j`` for the
        Lewis formula).
    T : float
        Time to maturity in years.  Must be positive.
    params : tuple of float
        Heston parameters ``(v0, kappa, theta, sigma, rho)``:

        - ``v0``    : initial instantaneous variance (> 0)
        - ``kappa`` : mean-reversion speed (> 0)
        - ``theta`` : long-run variance (> 0)
        - ``sigma`` : volatility of variance / vol-of-vol (> 0)
        - ``rho``   : spot–variance correlation (in (−1, 1))

    Returns
    -------
    complex or np.ndarray
        Characteristic function value(s) at ``u``.

    Notes
    -----
    **Branch cut** — Re(d) > 0 is enforced by negating ``d`` when Re(d) < 0.
    This is the Gatheral / "Little Trap" convention (Albrecher et al., 2007),
    which prevents discontinuous jumps in the integrand of the Lewis formula
    as ``u`` sweeps from 0 to umax.

    References
    ----------
    Heston, S. (1993). A closed-form solution for options with stochastic
    volatility. *Review of Financial Studies*, 6(2), 327–343.

    Gatheral, J. (2006). *The Volatility Surface*. Wiley Finance.

    Albrecher, H., Mayer, P., Schachermayer, W., & Teugels, J. (2007).
    The little Heston trap. *Wilmott Magazine*, 83–92.
    """
    v0, kappa, theta, sigma, rho = params

    # Affine coefficients of the Heston Riccati ODE
    alpha = -0.5 * (u**2 + 1j * u)
    beta  = kappa - rho * sigma * 1j * u
    gamma = 0.5 * sigma**2

    # Discriminant of the Riccati characteristic polynomial
    d = np.sqrt(beta**2 - 4.0 * alpha * gamma)

    # Enforce Re(d) > 0 (Gatheral branch convention)
    d_real = np.real(d)
    if np.ndim(d_real) > 0:
        # Array path
        d = np.where(d_real < 0, -d, d)
    elif d_real < 0:
        # Scalar path
        d = -d

    g = (beta - d) / (beta + d)
    exp_dt = np.exp(-d * T)

    # Riccati solution: variance coefficient B(u, T)
    B = (beta - d) / sigma**2 * (1.0 - exp_dt) / (1.0 - g * exp_dt)

    # Log-affine term A(u, T)
    log_term = np.log((1.0 - g * exp_dt) / (1.0 - g))
    A = (kappa * theta / sigma**2) * ((beta - d) * T - 2.0 * log_term)

    return np.exp(A + B * v0)


def heston_cf_and_grads(
    u: complex | NDArray,
    T: float,
    params: tuple[float, float, float, float, float],
) -> tuple:
    """Heston CF and its analytic gradients w.r.t. (v0, kappa, theta, sigma, rho).

    Differentiates through all Riccati-ODE intermediates analytically using the
    chain rule.  The gradient formulas are derived by implicit differentiation of
    ``d² = β² − 4αγ`` and explicit differentiation of ``B``, ``A``, and ``φ``.

    The branch convention (Re(d) > 0) is preserved: implicit differentiation of
    ``d² = β² − 4αγ`` yields ``∂d/∂p = (β·∂β/∂p − 2α·∂γ/∂p)/d``, which is
    branch-invariant (correct for both signs of the raw square root).

    Parameters
    ----------
    u : complex or array_like
        Frequency argument (same as for :func:`heston_cf`).
    T : float
        Time to maturity.
    params : tuple of float
        ``(v0, kappa, theta, sigma, rho)``.

    Returns
    -------
    phi, dphi_dv0, dphi_dkappa, dphi_dtheta, dphi_dsigma, dphi_drho
        CF value and five parameter gradients, all with the same shape as ``u``.

    Notes
    -----
    Used by :func:`~heston.calibration.jacobian_analytic` to supply exact
    Jacobians to :func:`~heston.calibration.calibrate_scipy` without finite
    differences.
    """
    v0, kappa, theta, sigma, rho = params

    # ------------------------------------------------------------------
    # CF intermediates  (same as heston_cf)
    # ------------------------------------------------------------------
    alpha = -0.5 * (u**2 + 1j * u)
    beta  = kappa - rho * sigma * 1j * u
    gamma = 0.5 * sigma**2

    d = np.sqrt(beta**2 - 4.0 * alpha * gamma)
    d_real = np.real(d)
    if np.ndim(d_real) > 0:
        d = np.where(d_real < 0, -d, d)
    elif d_real < 0:
        d = -d

    P   = beta - d
    Q   = beta + d
    g   = P / Q
    E   = np.exp(-d * T)
    N_B = 1.0 - E
    D_B = 1.0 - g * E
    F   = 1.0 - g
    B   = P * N_B / (sigma**2 * D_B)
    log_term = np.log(D_B / F)
    A   = (kappa * theta / sigma**2) * (P * T - 2.0 * log_term)

    phi = np.exp(A + B * v0)

    # ------------------------------------------------------------------
    # v0: A and B do not depend on v0
    # ------------------------------------------------------------------
    dphi_dv0 = B * phi

    # ------------------------------------------------------------------
    # theta: A = (κ/σ²)·θ·(P·T − 2·log_term) is linear in θ
    # ------------------------------------------------------------------
    dphi_dtheta = (A / theta) * phi

    # ------------------------------------------------------------------
    # Generic gradient helper
    # For parameter p with ∂β/∂p = beta_p and ∂γ/∂p = gamma_p:
    #
    #   d_p    = (β·β_p − 2α·γ_p) / d       (implicit differentiation)
    #   P_p    = β_p − d_p
    #   Q_p    = β_p + d_p
    #   g_p    = (P_p·Q − P·Q_p) / Q²
    #   E_p    = −T·E·d_p
    #   N_B_p  = −E_p  (= T·E·d_p)
    #   D_B_p  = −(g_p·E + g·E_p)
    #   F_p    = −g_p
    #   B_p    = B·(P_p/P + N_B_p/N_B − D_B_p/D_B) + B_extra
    #   L_p    = D_B_p/D_B − F_p/F
    #   A_p    = (κθ/σ²)·(P_p·T − 2·L_p) + A_extra
    # ------------------------------------------------------------------
    def _grad(beta_p, gamma_p, A_extra=0.0, B_extra=0.0):
        d_p   = (beta * beta_p - 2.0 * alpha * gamma_p) / d
        P_p   = beta_p - d_p
        Q_p   = beta_p + d_p
        g_p   = (P_p * Q - P * Q_p) / Q**2
        E_p   = -T * E * d_p
        N_B_p = -E_p
        D_B_p = -(g_p * E + g * E_p)
        F_p   = -g_p

        B_p = B * (P_p / P + N_B_p / N_B - D_B_p / D_B) + B_extra
        L_p = D_B_p / D_B - F_p / F
        A_p = (kappa * theta / sigma**2) * (P_p * T - 2.0 * L_p) + A_extra

        return (A_p + v0 * B_p) * phi

    # kappa: ∂β/∂κ = 1, ∂γ/∂κ = 0; A carries an explicit κ factor → extra A/κ
    dphi_dkappa = _grad(1.0, 0.0, A_extra=A / kappa)

    # sigma: ∂β/∂σ = −ρiu, ∂γ/∂σ = σ; A and B both carry 1/σ² → extra −2A/σ, −2B/σ
    dphi_dsigma = _grad(
        -rho * 1j * u, sigma,
        A_extra=-2.0 * A / sigma,
        B_extra=-2.0 * B / sigma,
    )

    # rho: ∂β/∂ρ = −σiu, ∂γ/∂ρ = 0
    dphi_drho = _grad(-sigma * 1j * u, 0.0)

    return phi, dphi_dv0, dphi_dkappa, dphi_dtheta, dphi_dsigma, dphi_drho
