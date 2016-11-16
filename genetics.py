from nnmath import *
from random import randint

# Super class of a gene
class Gene:
	fitness = 0
	genome = []

	def __init__(self):
		pass

	def mutate(self):
		pass

	def breed(self, gene):
		pass

	def encode(self):
		pass

	def evaluate(self):
		pass


class GeneticAlgorithm:
	def __init__(self, error, mutation_rate, obj, training_data, targets):
		self.obj = obj
		self.mutation_rate = mutation_rate
		self.error = error
		self.training_data = training_data
		self.targets = targets

	def population(self, size):
		# Use the factory method to create the population
		self.population = [self.obj() for i in range(size)]

	def evaluate(self):
		for gene in self.population:
			for data in self.training_data:
				(tag, input_vector) = data
				output = gene.evaluate(input_vector)

				target_vector = [0] * len(self.targets)
				for i, field in enumerate(self.targets):
					if field in tag:
						target_vector[i] = 1;

				gene.fitness = np.sum(np.multiply(output, target_vector))/len(target_vector)

		self.population = sorted(self.population, key=lambda gene: gene.fitness)

	def select(self):
		# Kill the least fit gene
		self.population = self.population[1:]

	def breed(self):
		offsprings = []
		for gene in self.population:
			# Breed with random gene from the population
			offspring = gene.breed(self.population[randint(0, len(self.population)-1)])

			offspring.mutate(self.mutation_rate)
			offsprings.append(offspring)
			gene.mutate(self.mutation_rate)

		self.population = self.population + offsprings

	def fittest(self):
		return self.population[-1]

	def evolve(self):
		return True if (1 - self.fittest().fitness) > self.error else False
