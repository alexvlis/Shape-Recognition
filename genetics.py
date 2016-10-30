from nnmath import *
from random import randint

# Base class definition of a gene
class Gene:
	def __init__(self):
		self.fitness = 0

	def mutate(self):
		pass

	def breed(self, gene):
		pass

	def encode(self):
		pass

	def fitness(self):
		return self.fitness


class GeneticAlgorithm:
	def __init__(self, obj, targets):
		self.cores = multiprocessing.cpu_count()
		self.obj = obj
		self.targets = targets

	def population(self, size):
		# Use the factory method to create the population
		self.population = [self.obj(self.targets) for i in range(size)]
	
	def evaluate(self, data):
		(input_vector, target) = data

		for nn in self.population:
			output = nn.feed_forward(input_vector)
			nn.fitness = euclidean(np.select(output, self.targets), target)

		self.population = sorted(self.population, key=lambda gene: gene.fitness())

	def select(self):
		# Kill the 5 weakest genes
		self.population = self.population[5:-1].copy()

	def breed(self):
		for gene in self.population:
			# Breed with random gene from the population
			offspring = gene.breed(self.population[randint(0, len(self.population))])
			offsprint.mutate()
			self.population.append(offspring)

	def best_gene(self):
		return self.population[-1]

	def evolve(self):
		return True
