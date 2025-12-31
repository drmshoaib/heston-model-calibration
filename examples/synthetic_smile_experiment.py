import numpy as np

from heston.pricing import heston_call_price
from heston.implied_vol import implied_vol_call, black_scholes_call
from heston.calibration import levenberg_marquardt
from utils_metrics import iv_error_metrics

rmse, mae, maxe = iv_error_metrics(S0, r, params_cal, market_data)

print("IV fit metrics:")
print(f"RMSE  : {rmse:.6f}")
print(f"MAE   : {mae:.6f}")
print(f"MaxErr: {maxe:.6f}")

# --------------------------------------------------
# Step 1: True Heston parameters (synthetic market)
# --------------------------------------------------
true_params = (
    0.04,   # v0
    1.5,    # kappa
    0.04,   # theta
    0.5,    # sigma
   -0.7     # rho
)

# --------------------------------------------------
# Step 2: Market environment
# --------------------------------------------------
S0 = 100.0
r  = 0.01
T  = 1.0

# --------------------------------------------------
# Step 3: Strike grid
# --------------------------------------------------
K = S0 * np.exp(np.linspace(-0.25, 0.25, 11))

# --------------------------------------------------
# Step 4: Generate synthetic implied-vol data
# --------------------------------------------------
data = []

for k in K:
    price = heston_call_price(S0, k, T, r, true_params)
    iv    = implied_vol_call(S0, k, T, r, price)

    # Black–Scholes vega (finite difference)
    eps = 1e-4
    vega = (
        black_scholes_call(S0, k, T, r, iv + eps)
        - black_scholes_call(S0, k, T, r, iv - eps)
    ) / (2 * eps)

    data.append((k, T, iv, vega))

# --------------------------------------------------
# Step 5: Initial guess and bounds
# --------------------------------------------------
initial_params = (
    0.02,
    0.5,
    0.02,
    0.3,
   -0.3
)

bounds = np.array([
    [1e-4, 0.5],   # v0
    [0.1,  5.0],   # kappa
    [1e-4, 0.5],   # theta
    [0.05, 2.0],   # sigma
    [-0.99, 0.99]  # rho
])

# --------------------------------------------------
# Step 6: Run calibration
# --------------------------------------------------
calibrated_params = levenberg_marquardt(
    initial_params,
    data,
    S0,
    r,
    bounds
)

# --------------------------------------------------
# Results
# --------------------------------------------------
print("True parameters:")
print(true_params)

print("\nCalibrated parameters:")
print(tuple(calibrated_params))
