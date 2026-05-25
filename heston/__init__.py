"""heston: Heston stochastic volatility model calibration library.

Public API
----------
``heston_cf``                 — Characteristic function (Gatheral branch convention)
``heston_cf_and_grads``       — CF + analytic gradients w.r.t. all 5 parameters
``heston_call_price``         — Fourier-based European call price (Lewis 2001)
``black_scholes_call``        — Black-Scholes call price
``implied_vol_call``          — Black-Scholes implied volatility (Brent inversion)
``calibrate_scipy``           — Calibration via scipy.optimize.least_squares  ← recommended
``multistart_calibrate``      — Sequential multi-start with reproducible seeds
``multistart_calibrate_parallel`` — Parallel multi-start via ProcessPoolExecutor
``levenberg_marquardt``       — Hand-rolled Levenberg–Marquardt (reference)
``objective``                 — Vega-weighted IV residual vector
``jacobian_analytic``         — Exact analytic Jacobian via CF chain rule
``enforce_bounds``            — Clip parameters to admissible range
``feller_condition``          — Diagnostic: 2·kappa·theta − sigma²
``DEFAULT_BOUNDS``            — Default parameter bounds array
"""

from .calibration import (
    DEFAULT_BOUNDS,
    calibrate_scipy,
    jacobian_analytic,
    levenberg_marquardt,
    multistart_calibrate,
    multistart_calibrate_parallel,
    objective,
)
from .cf import heston_cf, heston_cf_and_grads
from .implied_vol import black_scholes_call, implied_vol_call
from .pricing import heston_call_price
from .utils import enforce_bounds, feller_condition

__all__ = [
    "heston_cf",
    "heston_cf_and_grads",
    "heston_call_price",
    "black_scholes_call",
    "implied_vol_call",
    "calibrate_scipy",
    "multistart_calibrate",
    "multistart_calibrate_parallel",
    "levenberg_marquardt",
    "objective",
    "jacobian_analytic",
    "enforce_bounds",
    "feller_condition",
    "DEFAULT_BOUNDS",
]
