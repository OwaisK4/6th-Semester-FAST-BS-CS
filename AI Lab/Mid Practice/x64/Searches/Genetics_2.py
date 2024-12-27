import random
import string

TARGET = "Artificial Intelligence Lab"
POPULATION_SIZE = 70

def generate_random_string(length):
    return ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation + ' ') for _ in range(length))

def calculate_fitness_score(individual):
    return sum(1 for i, j in zip(individual, TARGET) if i != j)

def generate_initial_population(population_size):
    return [generate_random_string(len(TARGET)) for _ in range(population_size)]

def selection(population):
    return sorted(population, key=calculate_fitness_score)

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(TARGET) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutation(individual, mutation_rate):
    mutated_individual = ''
    for gene in individual:
        if random.random() < mutation_rate:
            mutated_individual += random.choice(string.ascii_letters + string.digits + string.punctuation + ' ')
        else:
            mutated_individual += gene
    return mutated_individual

def genetic_algorithm():
    population = generate_initial_population(POPULATION_SIZE)
    generation = 1
    while True:
        population = selection(population)
        if population[0] == TARGET:
            return population[0], generation
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = random.choices(population[:10], k=2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate=0.1)
            child2 = mutation(child2, mutation_rate=0.1)
            new_population.extend([child1, child2])
        population = new_population
        generation += 1

result, generation = genetic_algorithm()
print("Target string '{}' generated in generation {}".format(result, generation))