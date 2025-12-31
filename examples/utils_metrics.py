import numpy as np
from heston.pricing import heston_call_price
from heston.implied_vol import implied_vol_call

def iv_error_metrics(S0, r, params, market_data):
    errors = []
    for K, T, iv_mkt in market_data:
        price = heston_call_price(S0, K, T, r, params)
        iv_mod = implied_vol_call(S0, K, T, r, price)
        errors.append(iv_mod - iv_mkt)

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mae  = np.mean(np.abs(errors))
    maxe = np.max(np.abs(errors))
    return rmse, mae, maxe
