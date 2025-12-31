import numpy as np

def heston_cf(u, T, params):
    """
    Heston characteristic function phi(u; T) for log-price X_T = log S_T,
    under the risk-neutral measure with zero dividends.

    Parameters
    ----------
    u : complex or np.ndarray
        Complex frequency.
    T : float
        Time to maturity.
    params : tuple
        (v0, kappa, theta, sigma, rho)

    Returns
    -------
    complex or np.ndarray
        Characteristic function evaluated at u.
    """
    v0, kappa, theta, sigma, rho = params

    # Affine coefficients
    alpha = -0.5 * (u**2 + 1j * u)
    beta  = kappa - rho * sigma * 1j * u
    gamma = 0.5 * sigma**2

    # Discriminant
    d = np.sqrt(beta**2 - 4.0 * alpha * gamma)

    # Enforce Gatheral branch: Re(d) > 0
    d = np.where(np.real(d) < 0, -d, d)

    # g-function
    g = (beta - d) / (beta + d)

    # Exponentials
    exp_dt = np.exp(-d * T)

    # Riccati solutions
    B = (beta - d) / (sigma**2) * (1 - exp_dt) / (1 - g * exp_dt)

    log_term = np.log((1 - g * exp_dt) / (1 - g))

    A = (kappa * theta / sigma**2) * ((beta - d) * T - 2.0 * log_term)

    return np.exp(A + B * v0)
