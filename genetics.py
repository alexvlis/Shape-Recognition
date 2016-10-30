from nnmath import *
from random import randint

# Super class of a gene
class Gene:
	self.fitness = 0
	self.genome = 0

	def __init__(self):
		pass

	def mutate(self):
		pass

	def breed(self, gene):
		pass

	def encode(self):
		pass

	def fitness(self):
		return self.fitness

	def evaluate(self):
		pass


class GeneticAlgorithm:
	def __init__(self, generations, mutation_rate, obj, targets):
		self.obj = obj
		self.targets = targets
		self.generation = 0
		self.armageddon = generations
		self.mutation_rate = mutation_rate

	def population(self, size):
		# Use the factory method to create the population
		self.population = [self.obj(self.targets) for i in range(size)]

	def evaluate(self, data):
		(input_vector, target) = data

		for gene in self.population:
			output = gene.evaluate(input_vector)
			gene.fitness = euclidean(np.select(output, self.targets), target)

		self.population = sorted(self.population, key=lambda gene: gene.fitness())

	def select(self):
		# Kill the 5 least fit genes
		self.population = self.population[0:-5].copy()

	def breed(self):
		offsprings = []
		for gene in self.population:
			# Breed with random gene from the population
			offspring = gene.breed(self.population[randint(0, len(self.population))])
			offspring.mutate()
			offsprings.append(offspring)
			gene.mutate()

		self.population.append(offsprings)

	def fittest(self):
		return self.population[0]

	def evolve(self):
		return True if self.generation < self.armageddon else False
