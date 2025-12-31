import numpy as np
from pathlib import Path

from heston.pricing import heston_call_price
from heston.implied_vol import implied_vol_call
from heston.calibration import levenberg_marquardt
from math import sqrt
from scipy.stats import norm

def bs_vega(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * norm.pdf(d1) * np.sqrt(T)

# Parameter bounds (v0, kappa, theta, sigma, rho)
bounds = np.array([
    [1e-4,  1.0],    # v0
    [0.01,  5.0],    # kappa
    [1e-4,  1.0],    # theta
    [0.01,  2.0],    # sigma
    [-0.99, 0.99],   # rho
])

# -----------------------------
# CONFIG (edit if you want)
# -----------------------------
S0 = 100.0
r  = 0.01

# True parameters (ordering must match your code):
# params = (v0, kappa, theta, sigma, rho)
true_params = (0.04, 1.5, 0.04, 0.5, -0.7)

# Grid for synthetic surface
Ks = np.linspace(70, 130, 15)
Ts = np.array([0.5, 1.0, 2.0])

# Multi-start initial guesses (same ordering)
initial_guesses = [
    (0.05, 0.5, 0.20, 0.30, -0.30),
    (0.10, 1.0, 0.30, 0.60, -0.50),
    (0.02, 2.0, 0.10, 0.40, -0.80),
    (0.20, 0.8, 0.40, 0.70, -0.20),
    (0.01, 1.5, 0.05, 0.20, -0.90),
]

# Single-start initial guess (can be one of the above)
params0 = initial_guesses[0]

# Output .tex files (optional)
SAVE_TEX = True
OUTDIR = Path("tables")
OUTDIR.mkdir(exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def generate_market_data():
    """Return list of tuples (K, T, iv_mkt) generated from true_params."""
    data = []
    for T in Ts:
        for K in Ks:
            price = heston_call_price(S0, K, T, r, true_params)
            iv = implied_vol_call(S0, K, T, r, price)
            vega = bs_vega(S0, K, T, r, iv)
            data.append((float(K), float(T), float(iv), float(vega)))
    return data


def iv_errors(params, data):
    """Return np.array of (iv_model - iv_market) for each (K,T)."""
    errs = []
    for (K, T, iv_mkt, _) in data:
        price = heston_call_price(S0, K, T, r, params)
        iv_mod = implied_vol_call(S0, K, T, r, price)
        errs.append(iv_mod - iv_mkt)
    return np.array(errs, dtype=float)


def metrics_from_errors(errs):
    rmse = float(np.sqrt(np.mean(errs**2)))
    mae  = float(np.mean(np.abs(errs)))
    maxe = float(np.max(np.abs(errs)))
    return rmse, mae, maxe


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def latex_table_true_vs_cal(true_p, cal_p):
    # true_p and cal_p are (v0,kappa,theta,sigma,rho)
    v0_t, k_t, th_t, s_t, r_t = true_p
    v0_c, k_c, th_c, s_c, r_c = cal_p
    tex = r"""\begin{table}[h!]
\centering
\begin{tabular}{lccccc}
\hline
 & $\kappa$ & $\bar v$ & $\sigma$ & $\rho$ & $v_0$ \\
\hline
True parameters & """ + f"{fmt(k_t,2)} & {fmt(th_t,2)} & {fmt(s_t,2)} & ${fmt(r_t,2)}$ & {fmt(v0_t,2)}" + r""" \\
Calibrated parameters & """ + f"{fmt(k_c,2)} & {fmt(th_c,2)} & {fmt(s_c,2)} & ${fmt(r_c,2)}$ & {fmt(v0_c,2)}" + r""" \\
\hline
\end{tabular}
\caption{True and calibrated Heston parameters for the synthetic experiment.}
\label{tab:true_vs_cal}
\end{table}
"""
    return tex


def latex_table_metrics(rmse, mae, maxe):
    tex = r"""\begin{table}[h!]
\centering
\begin{tabular}{lccc}
\hline
Metric & RMSE$_\sigma$ & MAE$_\sigma$ & Max Error \\
\hline
Value & """ + f"{fmt(rmse,6)} & {fmt(mae,6)} & {fmt(maxe,6)}" + r""" \\
\hline
\end{tabular}
\caption{Implied volatility fit metrics for the synthetic calibration experiment.}
\label{tab:metrics}
\end{table}
"""
    return tex


def latex_table_multistart(rows):
    # rows: list of (name, rmse, converged_bool)
    body = ""
    for name, rmse, conv in rows:
        body += f"{name} & {fmt(rmse,6)} & {'Yes' if conv else 'No'} \\\\\n"
    tex = r"""\begin{table}[h!]
\centering
\begin{tabular}{lcc}
\hline
Initialisation & RMSE$_\sigma$ & Converged \\
\hline
""" + body + r"""\hline
\end{tabular}
\caption{Multi-start calibration results for the synthetic implied volatility surface.}
\label{tab:multistart}
\end{table}
"""
    return tex


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    market_data = generate_market_data()

    # Single-start calibration
    params_cal = levenberg_marquardt(params0, market_data, S0, r, bounds)

    errs = iv_errors(params_cal, market_data)
    rmse, mae, maxe = metrics_from_errors(errs)

    # Multi-start calibration
    ms_rows = []
    for i, p0 in enumerate(initial_guesses, start=1):
        try:
            p = levenberg_marquardt(p0, market_data, S0, r, bounds)
            e = iv_errors(p, market_data)
            rm, _, _ = metrics_from_errors(e)
            ms_rows.append((f"Start {i}", rm, True))
        except Exception:
            ms_rows.append((f"Start {i}", float("nan"), False))

    # Print results
    print("\n=== Calibrated params (v0,kappa,theta,sigma,rho) ===")
    print(params_cal)

    print("\n=== IV Fit Metrics ===")
    print(f"RMSE_sigma: {rmse:.10f}")
    print(f"MAE_sigma : {mae:.10f}")
    print(f"MaxErr    : {maxe:.10f}")

    # Produce LaTeX tables
    tex1 = latex_table_true_vs_cal(true_params, params_cal)
    tex2 = latex_table_metrics(rmse, mae, maxe)
    tex3 = latex_table_multistart(ms_rows)

    print("\n=== LaTeX: True vs Calibrated Parameters ===\n")
    print(tex1)

    print("\n=== LaTeX: Fit Metrics ===\n")
    print(tex2)

    print("\n=== LaTeX: Multi-start Summary ===\n")
    print(tex3)

    if SAVE_TEX:
        (OUTDIR / "table_true_vs_cal.tex").write_text(tex1, encoding="utf-8")
        (OUTDIR / "table_metrics.tex").write_text(tex2, encoding="utf-8")
        (OUTDIR / "table_multistart.tex").write_text(tex3, encoding="utf-8")
        print(f"\nSaved tables to: {OUTDIR.resolve()}")
