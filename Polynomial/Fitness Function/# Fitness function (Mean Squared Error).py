# Fitness function (Mean Squared Error)
import numpy as np
from sklearn.metrics import mean_squared_error


# Sample data generation
def generate_data(num_points, degree, noise_level=0.1):
    X = np.linspace(-1, 1, num_points)
    coefficients = np.random.randn(degree + 1)
    y = sum(c * (X ** i) for i, c in enumerate(coefficients)) + np.random.randn(num_points) * noise_level
    return X, y, coefficients

# Polynomial model
def polynomial_model(X, coefficients):
    return sum(c * (X ** i) for i, c in enumerate(coefficients))

# Fitness function (Mean Squared Error)
def fitness_function(coefficients, X, y):
    y_pred = polynomial_model(X, coefficients)
    mse = mean_squared_error(y, y_pred)
    return mse

# Bees Algorithm for polynomial regression
def bees_algorithm(X, y, degree, ns=50, ne=5, nb=10, nre=10, nrb=5, ngh=0.1, stlim=10, max_iter=100):
    # Initialize scout bees
    scout_bees = [np.random.randn(degree + 1) for _ in range(ns)]
    
    # Stagnation counter
    stagnation_counter = np.zeros(ns)
    
    best_solution = None
    best_fitness = float('inf')
    
    for iteration in range(max_iter):
        # Evaluate fitness of all scout bees
        fitness_values = [fitness_function(bee, X, y) for bee in scout_bees]
        
        # Rank the bees by fitness
        ranked_bees = sorted(zip(scout_bees, fitness_values), key=lambda x: x[1])
        
        # Select elite and best sites
        elite_sites = ranked_bees[:ne]
        best_sites = ranked_bees[ne:nb+ne]
        
        # Recruit bees for elite sites
        new_solutions = []
        for site, fit in elite_sites:
            for _ in range(nre):
                new_bee = site + np.random.uniform(-ngh, ngh, size=(degree + 1))
                new_solutions.append(new_bee)
        
        # Recruit bees for best sites
        for site, fit in best_sites:
            for _ in range(nrb):
                new_bee = site + np.random.uniform(-ngh, ngh, size=(degree + 1))
                new_solutions.append(new_bee)
        
        # Update the solutions based on their fitness
        new_fitness_values = [fitness_function(bee, X, y) for bee in new_solutions]
        combined_solutions = ranked_bees + list(zip(new_solutions, new_fitness_values))
        
        # Sort combined solutions by fitness
        combined_solutions.sort(key=lambda x: x[1])
        
        # Keep the best ns solutions as new scout bees
        scout_bees = [sol[0] for sol in combined_solutions[:ns]]
        
        # Update the best solution found so far
        if combined_solutions[0][1] < best_fitness:
            best_solution = combined_solutions[0][0]
            best_fitness = combined_solutions[0][1]
            stagnation_counter.fill(0)  # Reset stagnation counter
        else:
            stagnation_counter += 1
        
        # Check for stagnation and abandon sites
        for i in range(ns):
            if stagnation_counter[i] >= stlim:
                scout_bees[i] = np.random.randn(degree + 1)
                stagnation_counter[i] = 0
        
        print(f"Iteration {iteration+1}, Best Fitness: {best_fitness}")
    
    return best_solution, best_fitness

# Example usage
num_points = 100
degree = 3
X, y, true_coefficients = generate_data(num_points, degree)

# Run the Bees Algorithm
best_coefficients, best_mse = bees_algorithm(X, y, degree, ns=50, ne=5, nb=10, nre=10, nrb=5, ngh=0.1, stlim=10, max_iter=100)

print(f"True coefficients: {true_coefficients}")
print(f"Best coefficients: {best_coefficients}")
print(f"Best MSE: {best_mse}")
