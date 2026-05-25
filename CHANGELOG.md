# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.2.0] — 2026-05-25

### Added
- `heston_cf_and_grads()` in `heston/cf.py`: analytic gradients of the Heston
  characteristic function w.r.t. all five parameters (v0, κ, θ, σ, ρ), derived
  by implicit differentiation of `d² = β² − 4αγ` and explicit chain-rule
  differentiation of the Riccati-ODE intermediates (B, A, φ).  Branch-invariant:
  the Gatheral Re(d) > 0 convention is respected without additional logic.
  Matches finite-difference gradients to < 1×10⁻⁶ relative error.
- `jacobian_analytic()` in `heston/calibration.py`: exact Jacobian of the
  vega-weighted calibration residual vector, computed from `heston_cf_and_grads`.
  The characteristic function is evaluated once per unique maturity across all
  strikes (not once per data point).  No finite-difference evaluations.
- `calibrate_scipy(..., jac='analytic')`: `calibrate_scipy` now accepts
  `jac='analytic'` to use the exact Jacobian.  Passes synthetic-data convergence
  tests (cost < 1×10⁻⁸).
- `multistart_calibrate_parallel()` in `heston/calibration.py`: parallel
  multi-start via `concurrent.futures.ProcessPoolExecutor` with automatic
  fallback to sequential execution.  Uses `_calibration_worker` at module scope
  for correct pickling under Windows spawn semantics.
- Gauss-Legendre quadrature method in `heston_call_price(..., method='gl')`:
  n-point GL quadrature on [0, umax] using `numpy.polynomial.legendre.leggauss`.
  Numerically stable for any n.  Retained as a research alternative; the default
  trapezoidal rule remains recommended (better accuracy per evaluation for the
  oscillatory Lewis integrand).
- `multistart_calibrate_parallel` and `jacobian_analytic` added to
  `heston/__init__.py` public API.
- Three new tests in `tests/test_calibration.py`:
  `test_jacobian_analytic_shape`, `test_jacobian_analytic_matches_fd`,
  `test_calibrate_scipy_analytic_jac_converges`.

### Changed
- `jacobian_cs()` docstring updated with a clear warning that the complex-step
  approach is broken for the Lewis formula: the integral's natural O(1) imaginary
  parts overflow when divided by h = 1e-200.  Retained for reference only.
- `calibrate_scipy` docstring updated: replaced `jac='cs'` description with
  `jac='analytic'`.
- Module docstring of `calibration.py` updated to document `jacobian_analytic`
  and `multistart_calibrate_parallel`.

### Fixed
- Import chain: `calibration.py` now imports `heston_cf_and_grads`,
  `_make_integration_grid`, and `_trapezoid` from their respective modules.

---

## [1.1.0] — 2025-05-25

### Added
- `calibrate_scipy()` in `heston/calibration.py`: scipy.optimize.least_squares
  (TRF algorithm) as the recommended calibration backend. Returns a rich result
  dict with cost, nfev, convergence status, and wall-clock runtime.
- `multistart_calibrate()`: reproducible multi-start orchestration with
  configurable number of starts, a random seed, and pluggable backend.
- Vectorised strike pricing in `heston/pricing.py`: `heston_call_price` now
  accepts a scalar or 1-D array of strikes, computing the characteristic
  function once per maturity (significant speedup in calibration loops).
- `tests/` directory with pytest coverage for CF, pricing, IV inversion, and
  calibration (`test_cf.py`, `test_pricing.py`, `test_implied_vol.py`,
  `test_calibration.py`, `conftest.py`).
- `examples/benchmark_pricing.py`: measures scalar vs vectorised pricing
  throughput and single-start calibration runtime.
- `pyproject.toml`: package metadata, core and dev dependencies, pytest and
  ruff configuration.
- `requirements.txt`: pinned environment for reproducibility.
- `CHANGELOG.md`, `CITATION.cff`, `LICENSE` (MIT).
- Type hints (`from __future__ import annotations`) throughout all public modules.
- NumPy-style docstrings on all public functions.
- Python `logging` module replacing bare `print` calls in calibration modules.
- Improved `.gitignore` (notebooks, build artefacts, coverage, IDE files).
- `DEFAULT_BOUNDS` constant exported from `heston/calibration.py`.
- Populated `heston/__init__.py` with full public API.

### Changed
- `heston/implied_vol.py`: IV inversion switched from hand-written bisection to
  `scipy.optimize.brentq` (faster, more robust bracket handling).
  `black_scholes_call` now uses `scipy.stats.norm.cdf` instead of the erf
  identity for clarity.
- `heston/cf.py`: added scalar-safe branch enforcement (handles both scalar u
  and array u inputs). Documented Gatheral / "Little Trap" branch convention.
- `heston/pricing.py`: replaced deprecated `np.trapz` with a portable shim
  (`np.trapezoid` on NumPy ≥ 2.0, falling back to `np.trapz`). Extracted
  `_make_integration_grid()` helper. Vectorised integrand over K.
- `heston/calibration.py`: `objective()` now handles NaN IV values gracefully
  (large penalty instead of crashing). `levenberg_marquardt` increased default
  `max_iter` to 50, added convergence logging.
- `examples/run_calibration.py`: fully rewritten as a clean, working
  reproducible example using the scipy backend.
- `examples/synthetic_smile_experiment.py`: fixed broken script (iv_error_metrics
  called before variables were defined); restructured with proper `if __name__`
  guard; switched to multistart_calibrate.
- README.md: major overhaul — architecture diagram, results tables, parameter
  recovery note, limitations, performance notes, GitHub topics.

### Fixed
- `examples/run_calibration.py`: `levenberg_marquardt` called without the
  required `bounds` argument.
- `examples/synthetic_smile_experiment.py`: `iv_error_metrics` invoked at module
  level before `S0`, `r`, `params_cal`, and `market_data` were defined.
- `examples/utils_metrics.py`: wrong tuple length assumed (3 elements instead
  of 4) when unpacking market data.

---

## [1.0.0] — 2025-01-01

### Added
- Initial release: Heston stochastic volatility model calibration framework.
- `heston/cf.py`: characteristic function with Gatheral branch convention.
- `heston/pricing.py`: Lewis (2001) Fourier inversion for European call pricing.
- `heston/implied_vol.py`: Black-Scholes IV inversion by bisection.
- `heston/calibration.py`: Levenberg–Marquardt calibration with finite-difference
  Jacobian and vega weighting.
- `examples/synthetic_smile_experiment.py`: multi-start experiment against
  noiseless synthetic surface.
- `examples/report_tables.py`: LaTeX table generation.
- Residual heatmap, surface comparison, and smile fit diagnostic plots.
- Accompanying technical PDF.
