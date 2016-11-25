from nnmath import *
from random import randint

# Gene Super Class
class Gene:
	fitness = 0
	score = 0

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
		self.target_error = error
		self.training_data = training_data
		self.imglen = len(training_data[0][1]) # Use first image in set
		self.targets = targets
		self.popsize = 0
		self.error = 1

	def population(self, size):
		# Use the factory method to create the population
		self.population = [self.obj(self.imglen) for i in range(size)]
		self.popsize = size

	def evaluate(self):
		for gene in self.population:
			# Reset the score
			gene.score = 0
			for data in self.training_data:
				(tags, input_vector) = data

				output = gene.evaluate(input_vector)
				target = map(lambda x: int(x in tags), self.targets)
				eval_vector = (target == np.around(output)).astype(np.int)

				gene.score += np.sum(np.multiply(output, eval_vector))

			gene.fitness = gene.score/(len(eval_vector) * len(self.training_data))

		self.population = sorted(self.population, key=lambda gene: gene.fitness)
		self.error = 1 - self.fittest().fitness

	def select(self):
		# Keep the population size constant
		self.population = self.population[-self.popsize:]

	def breed(self):
		offsprings = []
		for gene in self.population:
			# Breed with random gene from the population
			offspring = gene.breed(self.population[randint(0, self.popsize-2)], self.mutation_rate)

			offsprings.append(offspring)

		self.population = self.population + offsprings

	def roulette(self):
		pass

	def fittest(self):
		return self.population[-1]

	def evolve(self):
		return True if self.error > self.target_error else False
