import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from tqdm import tqdm

def solve_decision_boundary(K=110.0, r=0.06, sigma=0.2, T=1.0, n=100):
    """
    Solves for the optimal exercise boundary of an American put option using the
    integral equation method described by Kim in "The Analytic Valuation of American Options".
    """
    # Model Setup
    dt = T / n  # Time step size
    s = np.linspace(0, T, n + 1)  # Time grid from 0 to T

    # Initialize the optimal exercise boundary array
    # B[i] will store B(s_i)
    B = np.zeros(n + 1)

    # Set the boundary condition at expiration (s=0, which is t=T)
    B[0] = K

    # Helper Functions
    def d1(S, K_strike, t, r_rate, vol):
        """ Standard Black-Scholes d1 function."""
        if t <= 1e-8: # Avoid division by zero for t=0
            return np.inf if S > K_strike else -np.inf
        return (np.log(S / K_strike) + (r_rate + 0.5 * vol**2) * t) / (vol * np.sqrt(t))

    def d2(S, K_strike, t, r_rate, vol):
        """Standard Black-Scholes d2 function."""
        return d1(S, K_strike, t, r_rate, vol) - vol * np.sqrt(t)

    def european_put_price(S, K_strike, t, r_rate, vol):
        """European put option price formula."""
        if t <= 1e-8:
            return max(0, K_strike - S)
        nd2 = norm.cdf(-d2(S, K_strike, t, r_rate, vol))
        nd1 = norm.cdf(-d1(S, K_strike, t, r_rate, vol))
        return K_strike * np.exp(-r_rate * t) * nd2 - S * nd1

    # Recursive Solver
    # This loop solves for B(s_i) for i = 1, 2, ..., n
    for i in tqdm(range(1, n + 1)):
        time_to_maturity = s[i]

        # Define the integral equation to solve for B[i]
        # This is a function of a single variable, `current_B`, which is our guess for B(s_i)
        def integral_equation(current_B):
            # The integral is approximated using the trapezoidal rule.
            # It is solved recursively using previously found boundary values B[0]...B[i-1].
            
            # Integrand from Equation (13)
            def integrand(B_s, s_val, B_xi, xi_val):
                time_diff = s_val - xi_val
                if time_diff <= 1e-8:
                    return 0
                
                term1 = r * K * np.exp(-r * time_diff) * norm.cdf(-d2(B_s, B_xi, time_diff, r, sigma))
                return term1

            # Numerically integrate from 0 to s_i using trapezoidal rule
            integral_sum = 0.0
            # Sum known parts of the integral from ξ=0 to ξ=s_{i-1}
            for j in range(1, i):
                val1 = integrand(current_B, time_to_maturity, B[j-1], s[j-1])
                val2 = integrand(current_B, time_to_maturity, B[j], s[j])
                integral_sum += 0.5 * (val1 + val2) * dt
            
            # The right-hand side of the main equation
            rhs = european_put_price(current_B, K, time_to_maturity, r, sigma) + integral_sum
            
            # The left-hand side of the main equation
            lhs = K - current_B
            
            # The equation that must equal zero for the root finder
            return lhs - rhs

        # Solve for B[i] using the previous value as an initial guess.
        # The boundary is non-increasing, so B[i-1] is a good starting point.
        initial_guess = B[i-1]
        B[i] = fsolve(integral_equation, initial_guess)[0]

    # Return the decision boundary found in time increasing
    return B[::-1]
