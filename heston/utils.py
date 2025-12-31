import numpy as np

def enforce_bounds(params, bounds):
    return np.clip(params, bounds[:,0], bounds[:,1])

def feller_condition(params):
    v0, kappa, theta, sigma, rho = params
    return 2*kappa*theta - sigma**2
