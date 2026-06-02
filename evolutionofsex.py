import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)
# Genome
N_GENOME = int(30)
N_NEIGHBORS = int(4)


class Genome:
    def __init__(self, length, ploidy):
        if ploidy == 2:
            self.chr1 = np.random.uniform(0, 1, length)
            self.chr2 = np.random.uniform(0, 1, length)
            self.ploidy = ploidy
        elif ploidy == 1:
            self.chr1 = np.random.uniform(0, 1, length)
            self.ploidy = ploidy

    def crossing_over(self):
        pass

    def get_fitness(self):
        tot_fitness = calculate_fitness(self.chr1, interaction_matrix, weight_matrix) + calculate_fitness(self.chr2, interaction_matrix, weight_matrix)
        return round(tot_fitness, 4)

# Fitness function
# Interaction matrix
interaction_matrix = np.zeros((N_GENOME, N_GENOME), dtype=int)
for i in range(N_GENOME):
    candidate_interactions = [j for j in range(N_GENOME) if j != i]
    spread = 1.0
    raw_probs = stats.norm.pdf(candidate_interactions, loc=i, scale=spread)
    probs = raw_probs / np.sum(raw_probs)
    interactions = np.random.choice(candidate_interactions, size=N_NEIGHBORS, p=probs)
    interaction_matrix[i, interactions] = 1

# Weight matrix
weight_matrix = (np.random.uniform(-1, 1, size=(N_GENOME, N_GENOME))) * interaction_matrix


# Calculate fitness
def calculate_fitness(genome, interact_mat, weight_mat):
    total_fit = 0.0
    for n in range(N_GENOME):
        interact_indices = np.where(interact_mat[n] == 1)[0]
        interact_sum = 0.0
        for m in interact_indices:
            interact_sum += weight_mat[n, m] * (genome[n] * genome[m])
        norm_fit = 1 / (1 + np.exp(-interact_sum))

        total_fit += norm_fit

    return total_fit

# Testing area

diploid_ex = Genome(N_GENOME, ploidy = 2)
example_genomes = [Genome(N_GENOME, 2) for _ in range(500)]
fitness_list = [g.get_fitness() for g in example_genomes]

plt.hist(fitness_list, bins=50)
plt.ylabel('Counts')
plt.xlabel('Fitness')
plt.show(block=True)