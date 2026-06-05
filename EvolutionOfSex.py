import random
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import heapq

np.random.seed(42)

# Parameters
N_POP_INIT_SIZE = int(600)
N_GENOME_SIZE = int(60)
N_NEIGHBORS = int(4)
MUT_RATE = 0.03
GENERATIONS = int(125)
TOURNAMENT_SIZE = 3
DIST_MAX = 2 * math.sqrt(N_GENOME_SIZE)


# Genome class
class Genome:
    def __init__(self, length, ploidy, sex, chr1=None, chr2=None):
        if ploidy == 2:
            if chr1 is None or chr2 is None:
                self.chr1 = np.random.uniform(0, 1, length)
                self.chr2 = np.random.uniform(0, 1, length)
            else:
                self.chr1 = chr1
                self.chr2 = chr2
            self.ploidy = ploidy
            self.is_sex = sex
        elif ploidy == 1:
            if chr1 is None:
                self.chr1 = np.random.uniform(0, 1, length)
            else:
                self.chr1 = chr1
            self.ploidy = ploidy
            self.is_sex = sex
        self.age = 0
        self.expressed_vector = np.average([self.chr1, self.chr2], axis=0)
        self.fitness = self.get_fitness()

    def crossing_over(self):
        pass

    def get_fitness(self):
        tot_fitness = calculate_fitness(self.expressed_vector, interaction_matrix, weight_matrix)
        return round(tot_fitness, 4)

    def mutate(self):
        sigma = 1.0
        for attr in ('chr1', 'chr2'):
            chrom = getattr(self, attr)
            mut_mask = np.random.rand(N_GENOME_SIZE) < MUT_RATE
            gauss_noise = np.random.normal(0, sigma, N_GENOME_SIZE)
            mutated_chr = np.where(mut_mask, chrom + gauss_noise, chrom)
            setattr(self, attr, np.clip(mutated_chr, -1.0, 1.0))
        self.expressed_vector = np.average([self.chr1, self.chr2], axis=0)
        self.fitness = self.get_fitness()


# Sexual repro
def sexual_reproduction(par1, par2):
    if (par1.is_sex or par2.is_sex) is False:
        raise ValueError("One or more of the parents are asexual")
    chrom1 = random.choice([par1.chr1, par1.chr2]).copy()
    chrom2 = random.choice([par2.chr1, par2.chr2]).copy()
    child = Genome(length=N_GENOME_SIZE, ploidy=2, sex=True, chr1=chrom1, chr2=chrom2)
    return child


# Calculate fitness
def calculate_fitness(chrom, interact_mat, weight_mat):
    total_fit = 0.0
    for n in range(N_GENOME_SIZE):
        interact_indices = np.where(interact_mat[n] == 1)[0]
        interact_sum = 0.0
        for m in interact_indices:
            interact_sum += weight_mat[n, m] * (chrom[n] * chrom[m])
        norm_fit = 1 / (1 + np.exp(-interact_sum))
        total_fit += norm_fit
        #total_fit += interact_sum

    return total_fit


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_distance(pop1, pop2):
    dist_parallel = np.linalg.norm(pop1.chr1 - pop2.chr1) + np.linalg.norm(pop1.chr2 - pop2.chr2)
    dist_crossed = np.linalg.norm(pop1.chr1 - pop2.chr2) + np.linalg.norm(pop1.chr2 - pop2.chr1)
    return min(dist_parallel, dist_crossed)


# Tourney select
def tournament_selection(_population: list):
    k_size = TOURNAMENT_SIZE
    _proceed_flag = bool(False)
    pool1 = random.sample(_population, k_size)
    p1 = max(pool1, key=lambda ind: ind.fitness)
    if p1.is_sex is False:
        return [p1]
    while _proceed_flag is False:
        pool2 = random.sample([p for p in _population if p.is_sex is True], k_size)
        dists = [calculate_distance(p1, p) for p in pool2]
        thresh = heapq.nsmallest(max(2, round(TOURNAMENT_SIZE * 0.15)), dists)
        distmask = [hit in thresh for hit in dists]
        pool2 = np.array(pool2)[np.array(distmask)]
        p2 = max(pool2, key=lambda ind: ind.fitness)
        if p1 != p2:
            _proceed_flag = bool(True)
            return [p1, p2]


# Interaction matrix
interaction_matrix = np.zeros((N_GENOME_SIZE, N_GENOME_SIZE), dtype=int)

for i in range(N_GENOME_SIZE):
    candidate_interactions = [j for j in range(N_GENOME_SIZE) if j != i]
    spread = N_GENOME_SIZE / 10
    raw_probs = stats.norm.pdf(candidate_interactions, loc=i, scale=spread)
    probs = raw_probs / np.sum(raw_probs)
    interactions = np.random.choice(candidate_interactions, size=N_NEIGHBORS, p=probs)
    interaction_matrix[i, interactions] = 1

# Weight matrix
weight_matrix = (np.random.uniform(-1, 1, size=(N_GENOME_SIZE, N_GENOME_SIZE))) * interaction_matrix

# Simulation loop
population_sex = [Genome(N_GENOME_SIZE, 2, sex=True) for _ in range(N_POP_INIT_SIZE)]
population_asex = [Genome(N_GENOME_SIZE, 2, sex=False) for _ in range(N_POP_INIT_SIZE)]
population = population_sex
fitness_over_time = []
highest_fitness_over_time = []
#for MUT_RATE in np.linspace(0.01,0.3,30):
for gener in range(GENERATIONS):

    # Selection and reproduction
    children = []
    if len(population) > 1:
        for _ in range(N_POP_INIT_SIZE):
            parents = tournament_selection(population)
            if len(parents) == 1:
                newborn = parents
                newborn[0].mutate()
                children.append(newborn[0])
            elif len(parents) == 2:
                newborn = sexual_reproduction(parents[0], parents[1])
                newborn.mutate()
                children.append(newborn)
    population = children

# Logging
    if len(population) > 0:
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        highest_fitness = max(ind.fitness for ind in population)
        highest_fitness_over_time.append(highest_fitness)
        fitness_over_time.append(avg_fitness)
        print(f"Generation {gener}: Pop size: {len(population)} | Highest Fitness: {highest_fitness} | Mean fitness: {avg_fitness}")
    else:
        print(f"Generation {gener}: Population went extinct")

# Testing area

plt.plot(fitness_over_time)
plt.plot(highest_fitness_over_time)
plt.ylabel('Fitness')
plt.xlabel('Gens')
plt.show(block=True)
#plt.savefig(f"C:/Users/Kevin/OneDrive/Pictures/PyCharm/MutRate_{round(MUT_RATE,2)}.png")