"""Synthetic implied-volatility surface calibration experiment.

Generates a noiseless synthetic surface from known Heston parameters,
calibrates using multi-start scipy.optimize.least_squares, and reports:

- parameter comparison (true vs calibrated)
- IV fit metrics (RMSE, MAE, max error)
- per-start convergence summary
- LaTeX tables saved to tables/

Usage (from project root)::

    python examples/synthetic_smile_experiment.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heston.calibration import multistart_calibrate
from heston.implied_vol import implied_vol_call
from heston.pricing import heston_call_price

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

S0, r = 100.0, 0.01
TRUE_PARAMS = (0.04, 1.5, 0.04, 0.5, -0.7)   # (v0, kappa, theta, sigma, rho)

Ks = np.linspace(70.0, 130.0, 15)
Ts = np.array([0.5, 1.0, 2.0])

BOUNDS = np.array([
    [1e-4, 1.0],
    [0.01, 5.0],
    [1e-4, 1.0],
    [0.01, 2.0],
    [-0.99, 0.99],
])

N_STARTS = 5
SEED     = 42
SAVE_TEX = True
OUTDIR   = Path("tables")


# ── Data generation ───────────────────────────────────────────────────────────

def bs_vega(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S0 * norm.pdf(d1) * np.sqrt(T))


def generate_market_data():
    """Return list of (K, T, iv_mkt, vega) from true parameters."""
    data = []
    for T in Ts:
        for K in Ks:
            price = heston_call_price(S0, K, T, r, TRUE_PARAMS)
            iv    = implied_vol_call(S0, K, T, r, price)
            vega  = bs_vega(S0, K, T, r, iv)
            data.append((float(K), float(T), float(iv), float(vega)))
    return data


# ── Diagnostics ───────────────────────────────────────────────────────────────

def iv_errors(params, data):
    errs = []
    for K, T, iv_mkt, _ in data:
        price  = heston_call_price(S0, K, T, r, params)
        iv_mod = implied_vol_call(S0, K, T, r, price)
        errs.append(iv_mod - iv_mkt)
    return np.array(errs, dtype=float)


def metrics(errs):
    return (
        float(np.sqrt(np.mean(errs**2))),
        float(np.mean(np.abs(errs))),
        float(np.max(np.abs(errs))),
    )


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


# ── LaTeX table generators ────────────────────────────────────────────────────

def latex_true_vs_cal(true_p, cal_p):
    v0_t, k_t, th_t, s_t, r_t = true_p
    v0_c, k_c, th_c, s_c, r_c = cal_p
    return (
        r"\begin{table}[h!]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{lccccc}" "\n"
        r"\hline" "\n"
        r" & $v_0$ & $\kappa$ & $\bar{v}$ & $\sigma$ & $\rho$ \\" "\n"
        r"\hline" "\n"
        f"True       & {fmt(v0_t,4)} & {fmt(k_t,2)} & {fmt(th_t,4)}"
        f" & {fmt(s_t,2)} & {fmt(r_t,2)}" r" \\" "\n"
        f"Calibrated & {fmt(v0_c,4)} & {fmt(k_c,2)} & {fmt(th_c,4)}"
        f" & {fmt(s_c,2)} & {fmt(r_c,2)}" r" \\" "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\caption{True and calibrated Heston parameters (synthetic experiment).}" "\n"
        r"\label{tab:true_vs_cal}" "\n"
        r"\end{table}" "\n"
    )


def latex_metrics(rmse, mae, maxe):
    return (
        r"\begin{table}[h!]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{lccc}" "\n"
        r"\hline" "\n"
        r"Metric & RMSE$_\sigma$ & MAE$_\sigma$ & Max Error \\" "\n"
        r"\hline" "\n"
        f"Value & {fmt(rmse,6)} & {fmt(mae,6)} & {fmt(maxe,6)}" r" \\" "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\caption{Implied volatility fit metrics for the synthetic calibration experiment.}" "\n"
        r"\label{tab:metrics}" "\n"
        r"\end{table}" "\n"
    )


def latex_multistart(rows):
    body = "".join(
        f"{name} & {fmt(rm,6)} & {str(nfev) if nfev is not None else '---'}"
        f" & {'Yes' if conv else 'No'}" r" \\" "\n"
        for name, rm, nfev, conv in rows
    )
    return (
        r"\begin{table}[h!]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{lccc}" "\n"
        r"\hline" "\n"
        r"Initialisation & RMSE$_\sigma$ & Fevals & Converged \\" "\n"
        r"\hline" "\n"
        + body
        + r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\caption{Multi-start calibration results for the synthetic IV surface.}" "\n"
        r"\label{tab:multistart}" "\n"
        r"\end{table}" "\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(
        "Generating synthetic surface: %d strikes × %d maturities = %d points...",
        len(Ks), len(Ts), len(Ks) * len(Ts),
    )
    market_data = generate_market_data()

    log.info("Running multi-start calibration (%d starts, seed=%d)...", N_STARTS, SEED)
    best, all_results = multistart_calibrate(
        market_data, S0, r, BOUNDS, n_starts=N_STARTS, seed=SEED
    )

    params_cal        = best["params"]
    errs              = iv_errors(params_cal, market_data)
    rmse, mae, maxe   = metrics(errs)

    # ── Parameter comparison ──────────────────────────────────────────────────
    names = ["v0", "kappa", "theta", "sigma", "rho"]
    print()
    print("=" * 60)
    print("Calibrated Parameters")
    print("=" * 60)
    print(f"{'Parameter':<10} {'True':>10} {'Calibrated':>12}")
    print("-" * 60)
    for n, t, c in zip(names, TRUE_PARAMS, params_cal):
        print(f"{n:<10} {t:>10.4f} {c:>12.4f}")

    # ── Fit metrics ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Implied Volatility Fit Metrics")
    print("=" * 60)
    print(f"RMSE      : {rmse:.4e}")
    print(f"MAE       : {mae:.4e}")
    print(f"Max Error : {maxe:.4e}")

    # ── Multi-start summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Multi-start Summary")
    print("=" * 60)
    print(f"{'Start':<8} {'RMSE':>12} {'Fevals':>8} {'Conv':>6}")
    print("-" * 60)
    ms_rows = []
    for res in all_results:
        if res["params"] is not None:
            e       = iv_errors(res["params"], market_data)
            rm, _, _ = metrics(e)
        else:
            rm = float("nan")
        nfev = res.get("nfev")
        conv = res["success"]
        label = f"Start {res['start']}"
        nfev_str = str(nfev) if nfev is not None else "---"
        print(f"{label:<8} {rm:>12.6f} {nfev_str:>8} {str(conv):>6}")
        ms_rows.append((label, rm, nfev, conv))

    # ── LaTeX output ──────────────────────────────────────────────────────────
    tex1 = latex_true_vs_cal(TRUE_PARAMS, params_cal)
    tex2 = latex_metrics(rmse, mae, maxe)
    tex3 = latex_multistart(ms_rows)

    if SAVE_TEX:
        OUTDIR.mkdir(exist_ok=True)
        (OUTDIR / "table_true_vs_cal.tex").write_text(tex1, encoding="utf-8")
        (OUTDIR / "table_metrics.tex").write_text(tex2, encoding="utf-8")
        (OUTDIR / "table_multistart.tex").write_text(tex3, encoding="utf-8")
        log.info("LaTeX tables saved to %s", OUTDIR.resolve())
