import random


def fitness(chromosome):
    return sum(chromosome)


def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected = random.choices(population, probabilities, k=2)
    return selected


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


def genetic_algorithm(initial_population, generations, mutation_rate):
    population = initial_population
    for generation in range(generations):
        fitness_values = [fitness(chromosome) for chromosome in population]

        parents = roulette_wheel_selection(population, fitness_values)

        offspring = [
            crossover(parents[0], parents[1]) for _ in range(len(population) // 2)
        ]
        offspring = [gene for sublist in offspring for gene in sublist]

        mutated_offspring = [
            mutate(chromosome, mutation_rate) for chromosome in offspring
        ]

        population = mutated_offspring

    best_chromosome = max(population, key=fitness)
    # print(population)
    return best_chromosome


initial_population = [
    [0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 1, 1],
]

generations = 50
mutation_rate = 0.01

best_solution = genetic_algorithm(initial_population, generations, mutation_rate)
print("Best solution:", best_solution)
print("Fitness:", fitness(best_solution))
