import numpy as np
from random import randint
import copy

class GAKill(Exception):
	def __init__(self, message):
		self.message = message

# Gene Super Class
class Gene:
	fitness = 0
	score = 0
	genotype = []
	cursor = 0

	def __init__(self):
		pass

	def encode(self):
		pass

	def decode(self):
		pass

	def evaluate(self):
		pass

	def mutate(self, rate):
		gen_len = len(self.genotype)

		# Select some random chromosomes
		idx = np.random.random_integers(0, gen_len-1, size=(1, round(rate*gen_len)))

		# Add a small -/+ number
		self.genotype[idx] += 0.01 * (2 * np.random.random_sample(1) - 1)

	def read_genotype(self, delta):
		chunk = self.genotype[self.cursor:self.cursor + delta]
		self.cursor += delta
		return chunk


class GeneticAlgorithm:
	popsize = 0
	error = 1
	epoch = 0
	armageddon = 0

	def __init__(self, epochs, mutation_rate, data, targets, obj, args):
		"""
		This contructor takes multiple parameters as well as the constructor
		for the population and an n-tuple for the arguments of the contructor.
		It is assumed the contructor knows how to decompose this.
		"""
		self.obj = obj
		self.args = args
		self.mutation_rate = mutation_rate
		self.training_data = data
		self.targets = targets
		self.armageddon = epochs

	def populate(self, size):
		# Use the object constructor to create the population
		self.population = [self.obj(self.args) for _ in range(size)]
		self.popsize = size

	def singleton(self):
		return self.obj(self.args, build=False)

	def evaluate(self):
		for gene in self.population:
			gene.evaluate(self.training_data, self.targets)

		self.population = sorted(self.population, key=lambda gene: gene.fitness)
		self.error = 1 - self.fittest().fitness  # Set the global error

	def crossover(self):
		# Create new population using roulette selection and breeding
		population = [self.breed(self.roulette(2)) for _ in range(self.popsize)]
		self.population = population

	def breed(self, parents):
		# Make a new gene
		offspring = self.singleton()

		# Determine cutting points
		length = parents[0].genotype.size - 1
		cuts = [randint(0, round(length/2)), randint(round(length/2), length)]

		# Perform 2-point crossover
		# TODO: Implement random order of breeding
		offspring.genotype = np.concatenate((parents[0].genotype[:cuts[0]], parents[1].genotype[cuts[0]:cuts[1]], parents[0].genotype[cuts[1]:]))

		offspring.mutate(self.mutation_rate)
		offspring.decode()

		return offspring

	def roulette(self, n):
		# Gather the fitnesses from the best 10 genes
		choice = self.population[-self.popsize/2:]
		#choice = self.population
		fitnesses = map(lambda x: x.fitness, choice)
		fitnesses /= np.sum(fitnesses) # Normalise

		return np.random.choice(choice, n, p=fitnesses)

	def fittest(self):
		return copy.deepcopy(self.population[-1])

	def evolve(self):
		return True if self.epoch < self.armageddon else False
