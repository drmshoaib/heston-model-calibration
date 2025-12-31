import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from heston.pricing import heston_call_price
from heston.implied_vol import implied_vol_call

# -----------------------------
# Experiment setup
# -----------------------------
S0 = 100.0
r  = 0.01

true_params = (0.04, 1.5, 0.04, 0.5, -0.7)
calibrated_params = (0.0104280129, 0.5431388391, 0.1397651576, 0.4588983637, -0.7242591535)

Ks = np.linspace(70, 130, 15)
Ts = np.array([0.5, 1.0, 2.0])

# -----------------------------
# Generate synthetic market data
# -----------------------------
market_data = {}
for T in Ts:
    for K in Ks:
        price = heston_call_price(S0, K, T, r, true_params)
        iv    = implied_vol_call(S0, K, T, r, price)
        market_data[(K, T)] = iv

# -----------------------------
# Compute residuals
# -----------------------------
residuals = np.zeros((len(Ts), len(Ks)))

for i, T in enumerate(Ts):
    for j, K in enumerate(Ks):
        price = heston_call_price(S0, K, T, r, calibrated_params)
        iv_mod = implied_vol_call(S0, K, T, r, price)
        iv_mkt = market_data[(K, T)]
        residuals[i, j] = iv_mod - iv_mkt

# -----------------------------
# Plot heatmap
# -----------------------------
plt.figure(figsize=(9, 5))

sns.heatmap(
    residuals,
    xticklabels=np.round(Ks, 1),
    yticklabels=Ts,
    cmap="coolwarm",
    center=0.0,
    cbar_kws={"label": "IV residual (model − market)"}
)

plt.title("Residual Heatmap: Implied Volatility Errors")
plt.xlabel("Strike")
plt.ylabel("Maturity")

plt.tight_layout()
plt.savefig("fig_3_residual_heatmap.png", dpi=300)
plt.show()
