import numpy as np
from numpy.polynomial import laguerre, polynomial
from scipy.interpolate import interp1d

from tqdm import tqdm

from thesis.functions.simulations import simulate_heston_paths, simulate_cev_paths, simulate_gbm_paths, discount_func

def run_lsmc(model, constants, K, M, N, dt, regression_degree, regression_poly):
    """
    Runs a single LSMC simulation and computes the option price and the decision boundary.
    """
    if model == 'gbm':
        S_0, r, sigma = constants
        stock_price_matrix = simulate_gbm_paths(S_0, r, sigma, N, M, dt)
    if model == 'cev':
        S_0, r, sigma, gamma,  = constants
        stock_price_matrix = simulate_cev_paths(S_0, r, sigma, gamma, N, M, dt)
    if model == 'heston':
        S_0, V_0, r, kappa, theta, sigma, rho = constants
        stock_price_matrix = simulate_heston_paths(S_0, V_0, r, kappa, theta, sigma, rho, N, M, dt)
    
    cashflow_matrix = cashflow_matrix_from_stock_prices(K, r, dt, regression_degree, regression_poly, stock_price_matrix)

    decision_boundary = decision_boundary_from_cashflow_matrix(K, stock_price_matrix, cashflow_matrix)

    price = price_from_cashflow_matrix(r, dt, cashflow_matrix)

    return price, decision_boundary

def average_lsmc(L, model, constants, K, M, N, dt, reg_degree, reg_poly):
    """
    Performs multiple independent LSMC simulations and computes average prices and decision boundaries.
    Applies interpolation to the decision boundary with zeros where exercise was not optimal for any path at a given time step
    Reduces Monte Carlo variance and provides more stable estimates.
    """
    prices = np.zeros(L)
    decision_boundaries = np.zeros((L, N))

    for l in tqdm(range(L)):
        price, decision_boundary = run_lsmc(model, constants, K, M, N, dt, reg_degree, reg_poly)
        prices[l] = price
        decision_boundaries[l,:] = decision_boundary

    ave_price = np.mean(prices)

    # Mask zeros in the decision boundaries arising in case of no exercise for any path at a given time step 
    # Compute the average decision boundary ignoring the zeros
    masked_boundaries = np.ma.masked_equal(decision_boundaries, 0)
    ave_decision_boundary =  np.ma.mean(masked_boundaries, axis=0).filled(0)

    # In case of still some zeros (L=1), interpolate
    if np.any(ave_decision_boundary == 0):
        # Find indices of non-zero entries
        non_zero_indices = ave_decision_boundary != 0
        # Find the time points of the non-zero entries
        time_grid = np.linspace(dt, int(dt*N), N)
        non_zero_times = time_grid[non_zero_indices]
        non_zero_decision_boundary = ave_decision_boundary[non_zero_indices]
        # Interpolate over zeros using linear or spline interpolation
        interpolator = interp1d(non_zero_times, non_zero_decision_boundary, kind='linear', fill_value="extrapolate")
        smooth_decision_boundary = interpolator(time_grid)
        ave_decision_boundary = smooth_decision_boundary
        print("Interpolated")

    return ave_price, ave_decision_boundary

def cashflow_matrix_from_stock_prices(K, r, dt, regression_degree, regression_poly, stock_price_matrix):
    """
    Applies the LSMC algorithm to determine the optimal exercise timing and hence build the cashflow matrix.
    
    Takes the simulated stock price paths, performs backward induction with polynomial regression to estimate
    continuation values and determine when American options should be exercised along each simulated path.
    """
    M, N_plus_1 = stock_price_matrix.shape
    N = N_plus_1 - 1
    cashflow_matrix = np.zeros((M, N)) #one less column since no cashflow at time 0
    cashflow_matrix[:,-1] = np.maximum(0, K - stock_price_matrix[:,-1]) #known payoff at T
    for n in range(N-1, 0, -1):
        X = stock_price_matrix[:,n] #stock prices at time t=n
        
        # Calculate Y - discounted actual future cashflows if not exercised at n
        Y = np.zeros(M)
        for m in range(M):   
            for j in range(n, N):
                if cashflow_matrix[m,j] > 0: # Find when it was exercised
                    # cashflow_matrix[m,j] occurs at time j+1
                    # We're at time n, so discount by (j+1-n) periods
                    Y[m] = discount_func(r, (j+1-n)*dt, cashflow_matrix[m,j])
                    break
        
        itm_mask = stock_price_matrix[:,n] < K #check which paths are in-the-money at time t=n
        
        if np.sum(itm_mask) > 0:  # Only regress if there are ITM paths
            X_itm = X[itm_mask] #select the rows of X and Y to be used for the regression
            Y_itm = Y[itm_mask]
            if regression_poly =='simple':
                coeffs = polynomial.polyfit(X_itm, Y_itm, deg=regression_degree) #calculate simple regression polynomial coefficients
                E_Y_X = polynomial.polyval(X, coeffs) #evaluate continuation expected vakues E(Y|X) using the regression for all X
            
            if regression_poly =='laguerre':
                lag_coeffs = laguerre.lagfit(X_itm, Y_itm, deg=regression_degree) #calculate Laguerre polynomial coefficients
                E_Y_X = laguerre.lagval(X, lag_coeffs) #evaluate continuation expected vakues E(Y|X) using the regression for all X

            exercise_now_mask = (np.add(-X, K) > E_Y_X)&itm_mask #check which paths are itm and should be exercised at t=n
            cashflow_matrix[:,n-1] = np.add(-X, K)*exercise_now_mask #fill in the cashflow at time n if itm and exercised at time n
            cashflow_matrix[exercise_now_mask, n:] = 0.0 #update the cashflow at att times >n to be 0 if exercised at n
        else:
            print("No in-the-money paths to perform regression.")

    cashflow_matrix = cashflow_matrix + 0.0
    return cashflow_matrix

def price_from_cashflow_matrix(r, dt, cashflow_matrix):
    """
    Calculates the option price by discounting cashflows to present value.
    
    Takes the cashflow matrix from LSMC and computes the option price by
    discounting each path's cashflow back to time zero and averaging across paths.
    """
    M, N = cashflow_matrix.shape
    path_present_values = np.zeros(M) # Initialize array to store present values for each path
    for m in range(M):
        for n in range(N-1):
            if cashflow_matrix[m, n] != 0:
                path_present_values[m] = discount_func(r, (n+1)*dt, cashflow_matrix[m, n])
                break
    # Return the average present value across all paths
    option_price = np.mean(path_present_values)
    return option_price

def decision_boundary_from_cashflow_matrix(K, stock_price_matrix, cashflow_matrix):
    """
    Extracts the early exercise decision boundary from LSMC results.
    
    Determines the critical stock prices at each time step that define the
    optimal exercise boundary for the American option.
    """
    M, N = cashflow_matrix.shape
    decision_boundary = np.zeros(N)
    
    # At maturity, boundary is the strike price
    decision_boundary[-1] = K
    
    # For each time step (except maturity)
    for n in range(N-1):
        # Find paths where option was exercised at time n+1
        exercised_mask = cashflow_matrix[:, n] > 0
        
        if np.sum(exercised_mask) > 0:
            # Get stock prices at time n+1 for exercised paths
            exercised_prices = stock_price_matrix[exercised_mask, n+1]
            # Boundary is the maximum price among exercised paths
            decision_boundary[n] = np.max(exercised_prices)
            #print(np.sum(not_exercised_mask), np.sum(stock_price_matrix[:,n+1] > decision_boundary[n]))

    return decision_boundary

def compute_mean_absolute_error(benchmark, approximation):
    """
    Computes the Mean Absolute Error between benchmark and approximated decision boundaries.
    
    Interpolates and aligns two decision boundary curves to calculate the MAE,
    providing a quantitative measure of approximation accuracy.
    """
    # Find the lengths
    N_benchmark = len(benchmark)
    N_approx = len(approximation)

    # Define time grids
    t_benchmark = np.linspace(0, 1, N_benchmark)
    dt = 1/N_approx
    t_approx = np.linspace(dt, 1, N_approx)

    # Trim the benchmark grid to fit the match the approximation
    mask = t_benchmark >= dt
    t_benchmark_trimmed = t_benchmark[mask]
    benchmark_trimmed = benchmark[mask]

    # Upsample the approximation curve onto the trimmed benchmark time grid using interpolation
    upsampled_approximation_trimmed = np.interp(t_benchmark_trimmed, t_approx, approximation)

    #Calculate and return the mean absolute error of the approximation
    return np.mean(np.abs(benchmark_trimmed - upsampled_approximation_trimmed))