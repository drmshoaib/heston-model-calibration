"""Generate README figures 4-7 for heston-model-calibration.

Figures produced (saved to project root):
  fig_4_smile_all_maturities.png      -- Calibrated vs market IV for T = 0.5, 1.0, 2.0 Y
  fig_5_parameter_sensitivity.png     -- Effect of +/-20 % perturbation on each Heston param
  fig_6_integration_convergence.png   -- Absolute price error vs N (trapz) and n (GL)
  fig_7_multistart_costs.png          -- Per-start cost: random initialisation -> convergence

Also prints Markdown table data ready to paste into README.md.

Usage (from project root):
    python examples/generate_readme_figures.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heston.calibration import multistart_calibrate, objective
from heston.implied_vol import implied_vol_call
from heston.pricing import heston_call_price

# ── Global style ──────────────────────────────────────────────────────────────

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "lines.linewidth": 1.8,
})

BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
ORANGE = "#F4A582"
PURPLE = "#762A83"
GREY   = "#888888"

OUTDIR = Path(__file__).resolve().parent.parent   # project root

# ── Shared configuration ──────────────────────────────────────────────────────

TRUE_PARAMS = (0.04, 1.5, 0.04, 0.5, -0.7)
S0, r = 100.0, 0.01
Ks   = np.linspace(70.0, 130.0, 15)
Ts   = np.array([0.5, 1.0, 2.0])
BOUNDS = np.array([
    [1e-4, 1.0], [0.01, 5.0], [1e-4, 1.0], [0.01, 2.0], [-0.99, 0.99],
])
PARAM_NAMES = ["v₀", "κ", "θ", "σ", "ρ"]
PARAM_KEYS  = ["v0", "kappa", "theta", "sigma", "rho"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bs_vega(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S0 * norm.pdf(d1) * np.sqrt(T))


def _make_data(params, strikes, T):
    data = []
    for K in strikes:
        price = heston_call_price(S0, K, T, r, params)
        iv    = implied_vol_call(S0, K, T, r, price)
        vega  = _bs_vega(S0, K, T, r, iv)
        data.append((float(K), float(T), float(iv), float(vega)))
    return data


def _iv_surface(params, strikes, maturities):
    """IV surface as (n_T, n_K) array."""
    out = np.full((len(maturities), len(strikes)), np.nan)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            price = heston_call_price(S0, K, T, r, params)
            out[i, j] = implied_vol_call(S0, K, T, r, price)
    return out


def _per_maturity_metrics(params_cal, iv_mkt):
    """RMSE, MAE, MaxErr in bps for each maturity."""
    iv_mod = _iv_surface(params_cal, Ks, Ts)
    rows   = []
    for i, T in enumerate(Ts):
        diff  = (iv_mod[i] - iv_mkt[i]) * 1e4   # convert to bps
        rmse  = float(np.sqrt(np.mean(diff**2)))
        mae   = float(np.mean(np.abs(diff)))
        maxe  = float(np.max(np.abs(diff)))
        rows.append((T, rmse, mae, maxe))
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 -- Run multi-start calibration (used by fig 4, fig 7, tables)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("Step 1: Multi-start calibration (5 starts, 45 data points)")
print("=" * 64)

market_data = []
for T in Ts:
    market_data.extend(_make_data(TRUE_PARAMS, Ks, T))

t_cal = time.perf_counter()
best, all_results = multistart_calibrate(
    market_data, S0, r, BOUNDS, n_starts=5, seed=42
)
t_cal = time.perf_counter() - t_cal
params_cal = best["params"]
print(f"  Calibration wall-clock time: {t_cal:.1f} s")
print(f"  Best cost: {best['cost']:.4e}  |  converged: {best['success']}")

# Compute initial costs for each start (re-derive random starting points)
rng = np.random.default_rng(42)
p0_list = [rng.random(5) * (BOUNDS[:, 1] - BOUNDS[:, 0]) + BOUNDS[:, 0]
           for _ in range(5)]
initial_costs = [
    float(0.5 * np.sum(objective(p0, market_data, S0, r)**2))
    for p0 in p0_list
]

# Per-maturity metrics
iv_mkt_surface = _iv_surface(TRUE_PARAMS, Ks, Ts)
pm_metrics     = _per_maturity_metrics(params_cal, iv_mkt_surface)

print("\n  Calibrated parameters:")
for name, true, cal in zip(PARAM_KEYS, TRUE_PARAMS, params_cal):
    print(f"    {name:>7}: true={true:.4f}  cal={cal:.4f}  diff={abs(true-cal):.2e}")

print("\n  Per-maturity IV metrics (RMSE / MAE / Max Error in basis points):")
for T_val, rmse, mae, maxe in pm_metrics:
    print(f"    T={T_val:.1f}Y  RMSE={rmse:.4e}  MAE={mae:.4e}  Max={maxe:.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 -- Integration convergence study (used by fig 6, table)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("Step 2: Integration convergence (K=100, T=1, ATM)")
print("=" * 64)

K_ref, T_ref = 100.0, 1.0

# Reference price: trapezoidal with N=100 000
u_ref    = np.linspace(1e-6, 100.0, 100_000)
from heston.cf import heston_cf
from numpy.polynomial.legendre import leggauss
try:
    _trap = np.trapezoid
except AttributeError:
    _trap = np.trapz

def _trapz_price(N, K=K_ref, T=T_ref):
    u   = np.linspace(1e-6, 100.0, N)
    phi = heston_cf(u - 0.5j, T, TRUE_PARAMS)
    ker = phi / (u**2 + 0.25)
    lmk = np.log(K / S0)
    itg = np.real(ker * np.exp(-1j * u * lmk))
    return float(S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * _trap(itg, u))

def _gl_price(n, K=K_ref, T=T_ref, umax=100.0):
    xi, wi = leggauss(n)
    u  = 0.5 * umax * (xi + 1.0)
    w  = 0.5 * umax * wi
    phi = heston_cf(u - 0.5j, T, TRUE_PARAMS)
    ker = phi / (u**2 + 0.25)
    lmk = np.log(K / S0)
    itg = np.real(ker * np.exp(-1j * u * lmk))
    return float(S0 - np.sqrt(S0 * K) * np.exp(-r * T) / np.pi * np.dot(w, itg))

price_ref = _trapz_price(100_000)
print(f"  Reference price (N=100 000): {price_ref:.8f}")

trapz_Ns  = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
gl_ns     = [4, 8, 16, 32, 64, 128, 256]

trapz_errs = [abs(_trapz_price(N) - price_ref) for N in trapz_Ns]
gl_errs    = [abs(_gl_price(n) - price_ref) for n in gl_ns]

print("\n  Trapezoidal rule:")
for N, e in zip(trapz_Ns, trapz_errs):
    print(f"    N={N:>6}  error={e:.3e}")
print("\n  Gauss-Legendre:")
for n, e in zip(gl_ns, gl_errs):
    print(f"    n={n:>4}  error={e:.3e}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 -- Calibrated smile across all three maturities
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("Generating fig_4_smile_all_maturities.png")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
fig.suptitle(
    "Calibrated Heston IV Smile vs Synthetic Market -- All Maturities",
    fontsize=13, fontweight="bold", y=1.01,
)

T_labels = ["T = 0.5 Y", "T = 1.0 Y", "T = 2.0 Y"]
moneyness = Ks / S0

for idx, (ax, T_val, label) in enumerate(zip(axes, Ts, T_labels)):
    iv_mkt = iv_mkt_surface[idx] * 100   # % units
    iv_cal = np.array([
        implied_vol_call(S0, K, T_val, r,
                         heston_call_price(S0, K, T_val, r, params_cal))
        for K in Ks
    ]) * 100

    ax.plot(moneyness, iv_mkt, "o", color=BLUE,
            markersize=6, label="Synthetic market", zorder=3)
    ax.plot(moneyness, iv_cal, "-", color=RED,
            linewidth=2.2, label="Calibrated model", zorder=2)

    residuals = iv_cal - iv_mkt
    ax2 = ax.twinx()
    ax2.bar(moneyness, residuals * 1e4, width=0.012,
            color=GREEN, alpha=0.5, label="Residual (bps)")
    ax2.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Residual (bps)", fontsize=9, color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN, labelsize=8)
    ax2.set_ylim(-2, 2)

    ax.set_title(label, fontweight="bold")
    ax.set_xlabel("Moneyness  K / S₀")
    ax.set_ylabel("Implied Volatility (%)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    if idx == 0:
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2,
                  loc="upper right", fontsize=9, framealpha=0.9)

fig.tight_layout()
path4 = OUTDIR / "fig_4_smile_all_maturities.png"
fig.savefig(path4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {path4.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 -- Parameter sensitivity (+/-20 % perturbation, T = 1 Y)
# ═════════════════════════════════════════════════════════════════════════════

print("Generating fig_5_parameter_sensitivity.png")

T_sens  = 1.0
Ks_sens = np.linspace(70.0, 130.0, 50)
mk_sens = Ks_sens / S0
DELTA   = 0.20   # +/-20 % perturbation

def _smile(params, T=T_sens):
    return np.array([
        implied_vol_call(S0, K, T, r, heston_call_price(S0, K, T, r, params))
        for K in Ks_sens
    ]) * 100

base_smile = _smile(TRUE_PARAMS)

param_effects = {
    "v₀  (initial variance)\nshifts smile level": (0, DELTA),
    "κ  (mean-reversion speed)\ncompresses term structure": (1, DELTA),
    "θ  (long-run variance)\nsets long-run smile level": (2, DELTA),
    "σ  (vol-of-vol)\ncontrols smile curvature": (3, DELTA),
    "ρ  (spot-vol correlation)\ndrives smile skew": (4, DELTA),
}

fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=True)
fig.suptitle(
    "Parameter Sensitivity: +/-20 % Perturbation of Each Heston Parameter  (T = 1 Y)",
    fontsize=12, fontweight="bold", y=1.02,
)

colors_hi = RED
colors_lo = BLUE

for ax, (title, (idx, delta)) in zip(axes, param_effects.items()):
    p_hi = list(TRUE_PARAMS)
    p_lo = list(TRUE_PARAMS)
    if idx == 4:   # rho is negative; +/-20% of |rho|
        p_hi[idx] = TRUE_PARAMS[idx] * (1 - delta)
        p_lo[idx] = TRUE_PARAMS[idx] * (1 + delta)
    else:
        p_hi[idx] = TRUE_PARAMS[idx] * (1 + delta)
        p_lo[idx] = TRUE_PARAMS[idx] * (1 - delta)

    smile_hi = _smile(tuple(p_hi))
    smile_lo = _smile(tuple(p_lo))

    ax.fill_between(mk_sens, smile_lo, smile_hi,
                    alpha=0.15, color=PURPLE, label="range")
    ax.plot(mk_sens, base_smile, "k-",  linewidth=2.2, label="Base")
    ax.plot(mk_sens, smile_hi,  "--",   color=RED,  linewidth=1.5,
            label=f"+20 %")
    ax.plot(mk_sens, smile_lo,  ":",    color=BLUE, linewidth=1.5,
            label=f"−20 %")

    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("K / S₀")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    if ax is axes[0]:
        ax.set_ylabel("Implied Volatility (%)")
    ax.legend(fontsize=8, loc="upper right")

fig.tight_layout()
path5 = OUTDIR / "fig_5_parameter_sensitivity.png"
fig.savefig(path5, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {path5.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6 -- Integration convergence
# ═════════════════════════════════════════════════════════════════════════════

print("Generating fig_6_integration_convergence.png")

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(trapz_Ns, trapz_errs, "o-", color=BLUE, linewidth=2,
          markersize=7, label="Trapezoidal rule (N nodes)")
ax.loglog(gl_ns,    gl_errs,    "s--", color=RED,  linewidth=2,
          markersize=7, label="Gauss-Legendre (n nodes)")

# Annotate key operating points
ax.annotate("N = 2000\n(default)",
            xy=(2000, trapz_errs[trapz_Ns.index(2000)]),
            xytext=(1500, 3e-6),
            arrowprops=dict(arrowstyle="->", color="k", lw=1.2),
            fontsize=9, color=BLUE)
ax.annotate("n = 64\n(default GL)",
            xy=(64, gl_errs[gl_ns.index(64)]),
            xytext=(30, 5e-4),
            arrowprops=dict(arrowstyle="->", color="k", lw=1.2),
            fontsize=9, color=RED)

ax.axhline(1e-4, color=GREY, linestyle=":", linewidth=1,
           label="10⁻⁴ (sub-basis-point IV)")
ax.axhline(1e-6, color=GREY, linestyle="--", linewidth=1,
           label="10⁻⁶ reference")

ax.set_xlabel("Number of integration nodes")
ax.set_ylabel("Absolute price error")
ax.set_title(
    "Integration Convergence: Trapezoidal vs Gauss-Legendre\n"
    "ATM European call  (S₀=K=100, T=1Y, r=1%)",
    fontweight="bold",
)
ax.legend(fontsize=10, loc="lower left")
ax.set_xlim(3, 20000)
ax.set_ylim(1e-9, 1e-1)
ax.grid(True, which="both", alpha=0.4)

fig.tight_layout()
path6 = OUTDIR / "fig_6_integration_convergence.png"
fig.savefig(path6, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {path6.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 7 -- Multi-start cost landscape
# ═════════════════════════════════════════════════════════════════════════════

print("Generating fig_7_multistart_costs.png")

n_starts = len(all_results)
starts   = [f"Start {res['start']}" for res in all_results]
final_costs = [res["cost"] for res in all_results]

fig, ax = plt.subplots(figsize=(9, 4.5))

y_pos   = np.arange(n_starts)
bar_h   = 0.35

bars_i = ax.barh(y_pos + bar_h / 2, initial_costs, bar_h,
                 color=ORANGE, edgecolor="k", linewidth=0.6,
                 label="Cost at random initialisation")
bars_f = ax.barh(y_pos - bar_h / 2, final_costs, bar_h,
                 color=BLUE, edgecolor="k", linewidth=0.6,
                 label="Cost after convergence")

# Annotate final costs
for i, (y, c) in enumerate(zip(y_pos, final_costs)):
    ax.text(c * 1.8, y - bar_h / 2, f"{c:.1e}",
            va="center", ha="left", fontsize=8.5, color=BLUE)

ax.set_xscale("log")
ax.set_yticks(y_pos)
ax.set_yticklabels(starts)
ax.set_xlabel("Objective cost  (0.5 · Σ residuals²)", labelpad=8)
ax.set_title(
    "Multi-start Calibration: Initialisation Cost vs Converged Cost\n"
    "5 random starts (seed = 42), 45 synthetic quotes, scipy TRF",
    fontweight="bold",
)
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(left=1e-20, right=max(initial_costs) * 10)
ax.axvline(1e-10, color=GREY, linewidth=0.8, linestyle="--",
           label="convergence criterion")
ax.invert_yaxis()
ax.grid(True, axis="x", which="both", alpha=0.4)
fig.tight_layout()

path7 = OUTDIR / "fig_7_multistart_costs.png"
fig.savefig(path7, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {path7.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Print Markdown table data for README
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 64)
print("MARKDOWN TABLE DATA FOR README")
print("=" * 64)

# Table A: per-maturity IV metrics
print("\n### Per-Maturity IV Fit Metrics")
print()
print("| Maturity | RMSE (bps) | MAE (bps) | Max error (bps) |")
print("|---|---|---|---|")
for T_val, rmse, mae, maxe in pm_metrics:
    print(f"| T = {T_val:.1f} Y | {rmse:.3e} | {mae:.3e} | {maxe:.3e} |")

# Table B: multi-start detailed results
print("\n### Multi-start Detailed Results")
print()
print("| Start | Final cost | Fevals | Converged | IV RMSE (bps) |")
print("|---|---|---|---|---|")
for res in all_results:
    if res["params"] is not None:
        iv_mod = _iv_surface(res["params"], Ks, Ts)
        diff   = (iv_mod - iv_mkt_surface) * 1e4
        rmse   = float(np.sqrt(np.mean(diff**2)))
    else:
        rmse = float("nan")
    print(f"| Start {res['start']} | {res['cost']:.2e} | "
          f"{res['nfev'] or '---'} | {'Yes' if res['success'] else 'No'} | "
          f"{rmse:.4e} |")

# Table C: integration accuracy
print("\n### Trapezoidal Rule Accuracy vs N")
print()
print("| N (nodes) | Absolute price error | Notes |")
print("|---|---|---|")
notes_trapz = {
    50:    "first-order accuracy regime",
    100:   "",
    200:   "",
    500:   "",
    1000:  "",
    2000:  "**default** -- sub-basis-point IV accuracy",
    5000:  "",
    10000: "",
}
for N, e in zip(trapz_Ns, trapz_errs):
    print(f"| {N:>6} | {e:.3e} | {notes_trapz.get(N, '')} |")

print("\n### Gauss-Legendre Accuracy vs n")
print()
print("| n (nodes) | Absolute price error | Notes |")
print("|---|---|---|")
notes_gl = {4: "", 8: "", 16: "", 32: "", 64: "**default** GL", 128: "", 256: ""}
for n, e in zip(gl_ns, gl_errs):
    print(f"| {n:>4} | {e:.3e} | {notes_gl.get(n, '')} |")

print("\n" + "=" * 64)
print("Done.")
