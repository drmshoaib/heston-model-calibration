import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
market_data = []
for T in Ts:
    for K in Ks:
        price = heston_call_price(S0, K, T, r, true_params)
        iv    = implied_vol_call(S0, K, T, r, price)
        market_data.append((K, T, iv))

# -----------------------------
# Build grids
# -----------------------------
K_grid, T_grid = np.meshgrid(Ks, Ts)
iv_mkt = np.zeros_like(K_grid)
iv_mod = np.zeros_like(K_grid)

data_dict = {(K, T): iv for (K, T, iv) in market_data}

for i, T in enumerate(Ts):
    for j, K in enumerate(Ks):
        iv_mkt[i, j] = data_dict[(K, T)]

        price = heston_call_price(S0, K, T, r, calibrated_params)
        iv_mod[i, j] = implied_vol_call(S0, K, T, r, price)

# -----------------------------
# Plot surfaces
# -----------------------------
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(K_grid, T_grid, iv_mkt, cmap="viridis", edgecolor="none")
ax1.set_title("Market (Synthetic) IV Surface")
ax1.set_xlabel("Strike")
ax1.set_ylabel("Maturity")
ax1.set_zlabel("Implied Volatility")

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(K_grid, T_grid, iv_mod, cmap="viridis", edgecolor="none")
ax2.set_title("Calibrated Heston IV Surface")
ax2.set_xlabel("Strike")
ax2.set_ylabel("Maturity")
ax2.set_zlabel("Implied Volatility")

plt.tight_layout()
plt.savefig("fig_2_market_vs_model_iv_surface.png", dpi=300)
plt.show()
plt.close()