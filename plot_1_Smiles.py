import numpy as np
import matplotlib.pyplot as plt

from heston.pricing import heston_call_price
from heston.implied_vol import implied_vol_call

# -----------------------------
# Market / experiment setup
# -----------------------------
S0 = 100.0
r  = 0.01
T  = 1.0

strikes = np.linspace(60, 140, 25)

true_params = (0.04, 1.5, 0.04, 0.5, -0.7)
calibrated_params = (0.0104280129, 0.5431388391, 0.1397651576, 0.4588983637, -0.7242591535)

# -----------------------------
# Compute implied vol smiles
# -----------------------------
iv_true = []
iv_cal  = []

for K in strikes:
    price_true = heston_call_price(S0, K, T, r, true_params)
    price_cal  = heston_call_price(S0, K, T, r, calibrated_params)

    iv_true.append(implied_vol_call(S0, K, T, r, price_true))
    iv_cal.append(implied_vol_call(S0, K, T, r, price_cal))

iv_true = np.array(iv_true)
iv_cal  = np.array(iv_cal)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(strikes, iv_true, 'o-', label='True (synthetic)')
plt.plot(strikes, iv_cal,  's--', label='Calibrated')
plt.xlabel('Strike')
plt.ylabel('Implied volatility')
plt.title('Heston Implied Volatility Smile (T = 1Y)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_1_heston_smile_T1Y.png", dpi=300)
plt.show()


