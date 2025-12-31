import numpy as np
from .cf import heston_cf

def heston_call_price(
    S0,
    K,
    T,
    r,
    params,
    N=2000,
    umax=100.0
):
    """
    European call option price under the Heston model using the
    Lewis (2001) Fourier inversion formula.

    Assumes zero dividends.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    params : tuple
        (v0, kappa, theta, sigma, rho)
    N : int
        Number of integration points.
    umax : float
        Upper integration bound.

    Returns
    -------
    float
        Call option price.
    """

    # Integration grid
    u = np.linspace(1e-6, umax, N)

    # Shifted characteristic function
    phi = heston_cf(u - 0.5j, T, params)

    # Lewis integrand
    integrand = np.real(
        np.exp(-1j * u * np.log(K / S0)) *
        phi / (u**2 + 0.25)
    )

    # Numerical integration
    integral = np.trapz(integrand, u)

    # Call price
    price = (
        S0
        - np.sqrt(S0 * K)
        * np.exp(-r * T)
        / np.pi
        * integral
    )

    return price
