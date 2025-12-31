import numpy as np
from heston.calibration import levenberg_marquardt
from heston.pricing import heston_call_price

# Dummy example data: (K, T, iv_mkt)
data = [
    (4000, 0.1, 0.20),
    (4000, 0.5, 0.25),
    (4000, 1.0, 0.22)
]

S0 = 4000
r = 0.01

params0 = np.array([0.04, 2.0, 0.04, 0.4, -0.6])

params_calibrated = levenberg_marquardt(params0, data, S0, r)
print("Calibrated params:", params_calibrated)
