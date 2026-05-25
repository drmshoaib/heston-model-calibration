"""Utility functions for parameter handling and model diagnostics."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def enforce_bounds(
    params: NDArray,
    bounds: NDArray,
) -> NDArray:
    """Clip a parameter vector to its admissible range.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector of length n.
    bounds : np.ndarray, shape (n, 2)
        Column 0: lower bounds.  Column 1: upper bounds.

    Returns
    -------
    np.ndarray
        Parameter vector clipped element-wise to ``[bounds[:, 0], bounds[:, 1]]``.
    """
    return np.clip(params, bounds[:, 0], bounds[:, 1])


def feller_condition(
    params: tuple[float, float, float, float, float],
) -> float:
    """Evaluate the Feller condition: ``2·kappa·theta − sigma²``.

    The Feller condition (value > 0) is sufficient to ensure the CIR variance
    process remains strictly positive.  It is not enforced during calibration
    by default (the calibrated surface fit is the primary criterion), but is
    useful as a post-calibration diagnostic.

    Parameters
    ----------
    params : tuple of float
        ``(v0, kappa, theta, sigma, rho)``.

    Returns
    -------
    float
        ``2·kappa·theta − sigma²``.  Positive → Feller condition satisfied.
    """
    _, kappa, theta, sigma, _ = params
    return 2.0 * kappa * theta - sigma**2
