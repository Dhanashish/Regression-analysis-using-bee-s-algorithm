# Benchmark Testing Polynomial MSE
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate synthetic data (Polynomial) for different functions
def generate_data(num_points, function='sin', noise_level=0.1):
    X = np.linspace(0, 10, num_points).reshape(-1, 1)
    if function == 'sin':
        y = np.sin(X).ravel() + np.random.randn(num_points) * noise_level
    elif function == 'log':
        y = np.log(X + 1).ravel() + np.random.randn(num_points) * noise_level
    elif function == 'tan':
        y = np.tan(X).ravel() + np.random.randn(num_points) * noise_level
    else:
        raise ValueError("Unknown function. Choose from 'sin', 'log', or 'tan'.")
    return X, y

# Polynomial model fitting
def polynomial_model(X, coefficients, degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    return np.dot(X_poly, coefficients)

# Fitness function (Mean Squared Error - MSE)
def fitness_function(coefficients, X, y, degree):
    try:
        y_pred = polynomial_model(X, coefficients, degree)
        mse = mean_squared_error(y, y_pred)
    except Exception as e:
        print(f"Error during polynomial fitting: {e}")
        mse = np.inf
    return mse

# Bees Algorithm for Polynomial Regression
def bees_algorithm(X, y, degree, num_coefficients, ns, ne, nb, nre, nrb, ngh, stlim, max_iter=100):
    # Initialize scout bees (random initial polynomial coefficients)
    scout_bees = [np.random.randn(num_coefficients) for _ in range(ns)]
    
    # Stagnation counter to track site abandonment
    stagnation_counter = np.zeros(ns)
    
    best_solution = None
    best_fitness = float('inf')
    
    for iteration in range(max_iter):
        # Evaluate fitness (MSE) for all scout bees
        fitness_values = [fitness_function(bee, X, y, degree) for bee in scout_bees]
        
        # Rank the bees by their fitness (lower MSE is better)
        ranked_bees = sorted(zip(scout_bees, fitness_values), key=lambda x: x[1])
        
        # Select elite and best sites
        elite_sites = ranked_bees[:ne]
        best_sites = ranked_bees[ne:nb+ne]
        
        # Recruit bees for elite sites
        new_solutions = []
        for site, fit in elite_sites:
            for _ in range(nre):
                new_bee = site + np.random.uniform(-ngh, ngh, size=num_coefficients)
                new_solutions.append(new_bee)
        
        # Recruit bees for best sites
        for site, fit in best_sites:
            for _ in range(nrb):
                new_bee = site + np.random.uniform(-ngh, ngh, size=num_coefficients)
                new_solutions.append(new_bee)
        
        # Update solutions with new bees
        new_fitness_values = [fitness_function(bee, X, y, degree) for bee in new_solutions]
        combined_solutions = ranked_bees + list(zip(new_solutions, new_fitness_values))
        
        # Sort combined solutions by fitness
        combined_solutions.sort(key=lambda x: x[1])
        
        # Keep the best ns solutions as new scout bees
        scout_bees = [sol[0] for sol in combined_solutions[:ns]]
        
        # Update best solution
        if combined_solutions[0][1] < best_fitness:
            best_solution = combined_solutions[0][0]
            best_fitness = combined_solutions[0][1]
            stagnation_counter.fill(0)  # Reset stagnation counter if improvement occurs
        else:
            stagnation_counter += 1
        
        # Site abandonment if stagnation occurs
        for i in range(ns):
            if stagnation_counter[i] >= stlim:
                scout_bees[i] = np.random.randn(num_coefficients)  # Re-initialize a new random bee
                stagnation_counter[i] = 0
        
        print(f"Iteration {iteration+1}, Best Fitness (MSE): {best_fitness}")
    
    return best_solution, best_fitness

# Benchmarking and testing Bees Algorithm with different parameters
def benchmark_bees_algorithm(X, y, degree, num_coefficients, param_combinations):
    results = []
    for params in param_combinations:
        ns, ne, nb, nre, nrb, ngh, stlim = params
        print(f"Running with params: ns={ns}, ne={ne}, nb={nb}, nre={nre}, nrb={nrb}, ngh={ngh}, stlim={stlim}")
        best_solution, best_mse = bees_algorithm(X, y, degree, num_coefficients, ns, ne, nb, nre, nrb, ngh, stlim)
        results.append((params, best_mse))
    return results

# Example usage and benchmarking
if __name__ == "__main__":
    # Data generation
    num_points = 100
    degree = 4  # Polynomial degree
    X, y = generate_data(num_points, function='sin')
    
    # Define possible parameters for benchmarking
    param_combinations = [
        (50, 5, 10, 10, 5, 0.1, 10),  # Set 1
        (30, 3, 7, 7, 3, 0.2, 15),    # Set 2
        (70, 10, 15, 15, 7, 0.05, 20), # Set 3
        (40, 4, 8, 8, 4, 0.15, 12),   # Set 4
    ]
    
    # Run benchmarking
    num_coefficients = degree + 1  # Number of polynomial coefficients
    results = benchmark_bees_algorithm(X, y, degree, num_coefficients, param_combinations)
    
    # Output results
    for params, mse in results:
        print(f"Params: {params} => Best MSE: {mse}")

    # Plot the results
    param_labels = [f"Set {i+1}" for i in range(len(param_combinations))]
    mse_values = [mse for _, mse in results]
    
    plt.bar(param_labels, mse_values)
    plt.ylabel('Best MSE')
    plt.title('Bees Algorithm Parameter Benchmarking')
    plt.show()
