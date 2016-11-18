from nnmath import *
from random import randint

# Gene Super Class
class Gene:
	fitness = 0
	score = 0
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
		self.popsize = 0

	def population(self, size):
		# Use the factory method to create the population
		self.population = [self.obj() for i in range(size)]
		self.popsize = size

	def evaluate(self):
		for gene in self.population:
			# Reset the score
			gene.score = 0
			for data in self.training_data:
				(tags, input_vector) = data
				output = gene.evaluate(input_vector)

				target = np.array([0] * len(self.targets))
				for i, field in enumerate(self.targets):
					if field in tags:
						target[i] = 1;

				print output
				eval_vector = np.array([0] * len(self.targets))
				eval_vector = (target == np.around(output)).astype(np.int)
				print eval_vector
				gene.score += np.sum(np.multiply(output, eval_vector))

			maxscore = len(eval_vector) * len(self.training_data)
			gene.fitness = gene.score/maxscore

		self.population = sorted(self.population, key=lambda gene: gene.fitness)

	def select(self):
		# Keep the population size constant
		self.population = self.population[-(self.popsize+1):-1]

	def breed(self):
		offsprings = []
		for gene in self.population:
			# Breed with random gene from the population
			offspring = gene.breed(self.population[randint(0, len(self.population)-1)], self.mutation_rate)

			offsprings.append(offspring)

		self.population = self.population + offsprings

	def fittest(self):
		return self.population[-1]

	def evolve(self):
		# TODO: More sophisticated error calculation
		return True if (1 - self.fittest().fitness) > self.error else False
