import numpy as np
import numpy.random as npr

def discount_func(r, t, X):
    """
    Applies conitnuous discounting for constant interest rates for present value calculations.
    """
    return X*np.exp(-r*t)

def simulate_gbm_paths(S_0, r, vol, N, M, dt):
    """
    Simulates stock price paths under Geometric Brownian Motion.
    
    Generates multiple sample paths for stock prices following the classic GBM model
    with constant drift and volatility parameters using the exact solution.
    """
    #Initialize the stock price array
    S = np.full((M, N+1), S_0).astype(np.float64)

    #Generate the standard normal random variables
    Z = npr.randn(M, N)

    drift = (r - 0.5*vol**2)*dt
    diffusion = vol*np.sqrt(dt)*Z    

    for n in range(1,N+1):
        S[:, n] = S[:, n - 1] * np.exp(drift + diffusion[:, n - 1])

        # Optional: prevent negative prices
        S[:, n] = np.maximum(S[:, n], 1e-8)
    return S

def simulate_cev_paths(S_0, r, sigma, gamma, N, M, dt):
    """
    Simulates stock price paths under Constant Elasticity of Variance model.
    
    Uses Euler-Maruyama discretization to simulate paths where volatility depends
    on the stock price level according to a power law.
    """
    #Initialize the stock price array
    S = np.full((M, N+1), S_0).astype(np.float64)

    # Generate all BM increments
    dW = np.random.normal(0, 1, size=(M, N)) * np.sqrt(dt)

    for n in range(1, N + 1):
        S[:, n] = S[:, n - 1] + r*S[:, n - 1]*dt + sigma*(S[:, n - 1]**gamma)*dW[:, n - 1]
        
        # Optional: prevent negative prices
        S[:, n] = np.maximum(S[:, n], 1e-8)
    return S

def simulate_heston_paths(S_0, V_0, r, kappa, theta, sigma, rho, N, M, dt):
    """
    Simulates stock price paths under the Heston stochastic volatility model.
    
    Uses Euler-Maruyama scheme with correlated Brownian motions to simulate both
    stock prices and their time-varying volatility processes.
    """
    # Initialize arrays
    V = np.full((M, N+1), V_0).astype(np.float64)
    S = np.full((M, N+1), S_0).astype(np.float64)
    sqrt_dt = np.sqrt(dt)

    # Generate the correlated standard normal random variables
    Z = np.random.normal(0, 1, size=(M, N, 2))
    dZ_1 = Z[:, :, 0]
    dZ_2 = rho * Z[:, :, 0] + np.sqrt(1 - rho ** 2) * Z[:, :, 1]

    for n in range(1, N + 1):
        sqrt_V_prev = np.sqrt(V[:, n - 1])
        S[:, n] = S[:, n - 1] * np.exp((r - 0.5 * V[:, n - 1]) * dt + sqrt_V_prev * sqrt_dt * dZ_1[:, n - 1])
        V[:, n] = V[:, n - 1] + kappa * (theta - V[:, n - 1]) * dt + sigma * sqrt_V_prev * sqrt_dt * dZ_2[:, n - 1]
        
        #Make sure variance is positive
        V[:, n] = np.maximum(V[:, n], 1e-8)
        # Optional: prevent negative prices
        S[:, n] = np.maximum(S[:, n], 1e-8)
    return S