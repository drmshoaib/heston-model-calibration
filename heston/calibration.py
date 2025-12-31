import numpy as np
from .implied_vol import implied_vol_call
from .pricing import heston_call_price
from .utils import enforce_bounds

def objective(params, data, S0, r):
    res = []
    for (K, T, iv_mkt, vega) in data:
        price = heston_call_price(S0, K, T, r, params)
        iv_model = implied_vol_call(S0, K, T, r, price)
        res.append((iv_model - iv_mkt) / vega)
    return np.array(res)

def jacobian(params, data, S0, r, h=1e-4):
    n = len(params)
    m = len(data)
    J = np.zeros((m, n))
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        r_plus  = objective(params + h * e, data, S0, r)
        r_minus = objective(params - h * e, data, S0, r)
        J[:, j] = (r_plus - r_minus) / (2 * h)
    return J

def levenberg_marquardt(
    params0,
    data,
    S0,
    r,
    bounds,
    max_iter=20,
    lambda0=1e-3,
    tol=1e-6
):
    params = np.array(params0, dtype=float)
    lam = lambda0

    for _ in range(max_iter):
        res = objective(params, data, S0, r)
        J = jacobian(params, data, S0, r)

        A = J.T @ J + lam * np.eye(len(params))
        g = J.T @ res
        step = np.linalg.solve(A, g)

        new_params = enforce_bounds(params - step, bounds)

        if np.linalg.norm(step) < tol:
            break

        if np.sum(res**2) > np.sum(objective(new_params, data, S0, r)**2):
            params = new_params
            lam *= 0.5
        else:
            lam *= 2.0

    return params
