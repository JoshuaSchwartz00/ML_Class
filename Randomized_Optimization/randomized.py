import matplotlib.pyplot as plt
import numpy as np

import six
import sys
sys.modules["sklearn.externals.six"] = six
import mlrose

from sklearn.datasets import load_breast_cancer

#best sim anneal problem
#four peaks
def four_peaks():
    fitness = mlrose.FourPeaks()
    problem = mlrose.DiscreteOpt(10, fitness)
    
    init_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    return problem, init_state

#best mimic problem
#knapsack
def knapsack(items):
    weights = [i for i in range(1, items+1)]
    values = [i for i in range(1, items+1)]
    
    fitness = mlrose.Knapsack(weights, values, max_weight_pct=1.0)
    problem = mlrose.DiscreteOpt(items, fitness)
    
    arr = [0]*items
    init_state = np.array(arr)
    
    return problem, init_state

#n-queens
def n_queens():
    fitness = mlrose.Queens()
    problem = mlrose.DiscreteOpt(8, fitness, maximize=False, max_val=8)
    
    init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    
    return problem, init_state

#best genetic algorithm problem
#traveling salesman
def TSL(randomize=False, num_points=10):
    if randomize:
        x_coords = np.random.randint(0, 100, num_points)
        y_coords = np.random.randint(0, 100, num_points)
        
        coords_list = [(x, y) for x, y in zip(x_coords, y_coords)]
    else: #an octagon where the min distance is 4+4*sqrt(2) or about 9.68
        coords_list = [(1, 0), (2, 0), (0, 2), (2, 3), (1, 3), (3, 1), (0, 1), (3, 2)]
        num_points = 8
    
    fitness = mlrose.TravellingSales(coords = coords_list)
    problem = mlrose.TSPOpt(num_points, fitness_fn=fitness)
    
    arr = [x for x in range(num_points)]
    init_state = np.array(arr)
    
    return problem, init_state

#graphing
def graph_data(fitnesses, title, ylabel="Fitness"):
    plt.plot(fitnesses)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.show()
    plt.savefig(f"{title}.png")
    plt.clf()

#algorithms
def randomized_hill_climbing(problem, init_state, name, seed):    
    _, _, fitnesses = mlrose.random_hill_climb(problem, max_attempts=40, max_iters=100, restarts=10, init_state=init_state, curve=True, random_state=seed)
    
    title = f"Random Hill Climb Fitness for {name} Problem"
    graph_data(fitnesses, title)

def simulated_annealing(problem, init_state, name, seed):
    schedule = mlrose.ExpDecay()
        
    _, _, fitnesses = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=40, max_iters=100, init_state=init_state, curve=True, random_state=seed)
            
    title = f"Simulated Annealing Fitness for {name} Problem"
    graph_data(fitnesses, title)

def genetic(problem, name, seed):    
    _, _, fitnesses = mlrose.genetic_alg(problem, pop_size=1000, mutation_prob=.17, max_attempts=40, max_iters=100, curve=True, random_state=seed)
    
    title = f"Genetic Algorithm Fitness for {name} Problem"
    graph_data(fitnesses, title)

def mimic(problem, name, seed):
    _, _, fitnesses = mlrose.mimic(problem, pop_size=200, max_attempts=40, max_iters=100, curve=True, random_state=seed)
        
    title = f"MIMIC Fitness for {name} Problem"
    graph_data(fitnesses, title)
    
def neural_network(name, seed, X, y, algorithm="Random Hill Climb"):
    if algorithm == "Random Hill Climb":
        model = mlrose.NeuralNetwork(hidden_nodes=[100, 100], max_iters=1000, learning_rate=.001, early_stopping=True, \
                        restarts=10, max_attempts=40, random_state=seed, curve=True)
    elif algorithm == "Simulated Annealing":
        schedule = mlrose.ExpDecay()
        model = mlrose.NeuralNetwork(hidden_nodes=[100, 100], algorithm="simulated_annealing", max_iters=1000, learning_rate=.001, early_stopping=True, \
                        schedule=schedule, max_attempts=40, random_state=seed, curve=True)
    elif algorithm == "Genetic Algorithm":
        model = mlrose.NeuralNetwork(hidden_nodes=[100, 100], algorithm="genetic_alg", max_iters=1000, learning_rate=.001, early_stopping=True, \
                        mutation_prob=.17, max_attempts=40, random_state=seed, curve=True)
        
    model.fit(X, y)
    
    fitnesses = model.fitness_curve
    
    title = f"{algorithm} Fitness for {name} Problem"
    graph_data(fitnesses, title)
    
if __name__ == "__main__":
    seed = 1
    #seed = np.random.seed()
    
    #number of items in knapsack
    num_items = 20
    
    #randomizes coordinates for TSL
    randomize_coords = True
    
    #number of random points generated for TSL
    num_points = 20
    
    print("Four Peaks")
    #four peaks
    name = "Four Peaks"
    problem, init_state = four_peaks()
    
    randomized_hill_climbing(problem, init_state, name, seed)
    simulated_annealing(problem, init_state, name, seed)
    genetic(problem, name, seed)
    mimic(problem, name, seed)
    
    print("Knapsack")
    #knapsack
    name = "Knapsack"
    problem, init_state = knapsack(num_items)
    
    randomized_hill_climbing(problem, init_state, name, seed)
    simulated_annealing(problem, init_state, name, seed)
    genetic(problem, name, seed)
    mimic(problem, name, seed)
    
    print("TSL")
    #TSL
    name = "TSL"
    problem, init_state = TSL(randomize=randomize_coords, num_points=num_points)
    
    randomized_hill_climbing(problem, init_state, name, seed)
    simulated_annealing(problem, init_state, name, seed)
    genetic(problem, name, seed)
    mimic(problem, name, seed)
    
    #NN
    data = load_breast_cancer(return_X_y=True)
    X, y = data[0], data[1]
    name = "Neural Network"
    
    neural_network(name, seed, X, y, algorithm="Random Hill Climb")
    neural_network(name, seed, X, y, algorithm="Simulated Annealing")
    #neural_network(name, seed, X, y, algorithm="Genetic Algorithm")