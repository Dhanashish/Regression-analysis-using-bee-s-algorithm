# Fitness Function MAE for Log x
import numpy as np
import matplotlib.pyplot as plt

# Function Definitions
def log_func(x):
    return np.log(x)

# Fitness Function: Mean Absolute Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Polynomial Model
def poly_model(x, coeffs):
    return np.polyval(coeffs[::-1], x)  # Reverse order for numpy polyval

x_range = np.linspace(0.1, 2, 200)

# Bees Algorithm for regression
def bees_algorithm(func, x_range, degree, ns, ne, nb, nre, nrb, ngh, stlim):
    # Initial population (random coefficients for polynomial)
    population = [np.random.rand(degree + 1) - 0.5 for _ in range(ns)]
    fitness = [mae(func(x_range), poly_model(x_range, ind)) for ind in population]
    
    best_fitness_history = []
    
    # Optimization loop
    for gen in range(100):  # Number of generations
        # Sort by fitness and select the best
        indices = np.argsort(fitness)
        population = [population[i] for i in indices]
        fitness = [fitness[i] for i in indices]

        # Elite and best sites
        elites = population[:ne]
        bests = population[ne:ne+nb]

        # Recruitment for elite and best sites
        for i in range(ne):
            for _ in range(nre):
                candidate = elites[i] + np.random.randn(degree + 1) * ngh
                candidate_fitness = mae(func(x_range), poly_model(x_range, candidate))
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness

        for i in range(nb):
            for _ in range(nrb):
                candidate = bests[i] + np.random.randn(degree + 1) * ngh
                candidate_fitness = mae(func(x_range), poly_model(x_range, candidate))
                if candidate_fitness < fitness[ne + i]:
                    population[ne + i] = candidate
                    fitness[ne + i] = candidate_fitness

        # Track best fitness
        best_fitness_history.append(fitness[0])

        # Reduce neighborhood size
        ngh *= 0.95

        # Check for stagnation and reset if necessary
        if len(best_fitness_history) > stlim and best_fitness_history[-1] == best_fitness_history[-stlim]:
            population = [np.random.rand(degree + 1) - 0.5 for _ in range(ns)]
            fitness = [mae(func(x_range), poly_model(x_range, ind)) for ind in population]
            ngh = 0.1  # Reset neighborhood size

    return population[0], best_fitness_history

# Hypothetical parameter ranges
degrees = range(3, 6)  # Polynomial degrees to test
ns_values = [30, 50, 100]  # Different values for number of scout bees
ne_values = [5, 10, 15]  # Values for number of elite sites
nb_values = [5, 10, 15]  # Values for number of best sites
nre_values = [3, 5, 10]  # Recruited bees for elite sites
nrb_values = [2, 3, 5]  # Recruited bees for remaining best sites
ngh_values = [0.1, 0.01]  # Initial neighborhood sizes
stlim_values = [10, 20, 30]  # Stagnation limits

best_setup = None
best_fit = np.inf

# Loop over all combinations (simple grid search approach)
for degree in degrees:
    for ns in ns_values:
        for ne in ne_values:
            for nb in nb_values:
                for nre in nre_values:
                    for nrb in nrb_values:
                        for ngh in ngh_values:
                            for stlim in stlim_values:
                                coeffs, fitness_history = bees_algorithm(log_func, x_range, degree, ns, ne, nb, nre, nrb, ngh, stlim)
                                final_fitness = fitness_history[-1]
                                if final_fitness < best_fit:
                                    best_fit = final_fitness
                                    best_setup = (degree, ns, ne, nb, nre, nrb, ngh, stlim)

print("Best setup:", best_setup)

# Run the Bees Algorithm
best_coeffs, fitness_history = bees_algorithm(log_func, x_range, degree, ns, ne, nb, nre, nrb, ngh, stlim)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(x_range, log_func(x_range), label='log(x)', color='blue')
plt.plot(x_range, poly_model(x_range, best_coeffs), label='Fitted Model', color='red')
plt.title('Log(x) and Fitted Polynomial')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(fitness_history, color='green')
plt.title('Fitness History (MAE)')
plt.xlabel('Generation')
plt.ylabel('MAE')
plt.show()