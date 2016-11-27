import copy
from random import randint

# Gene Super Class
class Gene:
	fitness = 0
	score = 0
	genotype = ""

	def __init__(self):
		pass

	def encode(self):
		pass

	def decode(self):
		pass

	def mutate(self, rate):
		gen_len = len(self.genotype)

		for _ in round(rate * gen_len)
			idx = randint(0, gen_len) # Select a random bit
			# Flip the bit
			genotype[idx] = '0' if genotype[randint(0, gen_len)] == '1' else '1'

	def evaluate(self):
		pass


class GeneticAlgorithm:
	def __init__(self, error, mutation_rate, data, targets, obj, args):
		self.obj = obj
		self.mutation_rate = mutation_rate
		self.target_error = error
		self.training_data = data
		self.targets = targets
		self.popsize = 0
		self.error = 1

	def population(self, size):
		# Use the factory method to create the population
		self.population = [self.obj(args) for i in range(size)]
		self.popsize = size

	def evaluate(self):
		for gene in self.population:
			# Reset the score
			gene.score = 0
			for data in self.training_data:
				(tags, input_vector) = data
				print tags

				output = gene.evaluate(input_vector)
				print output
				target = map(lambda x: int(x in tags), self.targets)
				eval_vector = (target == np.around(output)).astype(np.int)

				scorer = [0] * len(output)
				for i, value in enumerate(target):
					if value == 0:
						scorer[i] = 1 - output[i]
					else:
						scorer[i] = output[i]

				gene.score += np.dot(scorer, eval_vector)

			gene.fitness = gene.score/(len(eval_vector) * len(self.training_data))
			print "------------------------------------------------------------"

		self.population = sorted(self.population, key=lambda gene: gene.fitness)
		self.error = 1 - self.fittest().fitness


	def crossover(self):
		offsprings = []
		for gene in self.population:
			# Breed with random gene from the population
			offspring = gene.breed(self.population[randint(0, self.popsize-2)], self.mutation_rate)

			offsprings.append(offspring)

		self.population = self.population + offsprings

	def breed(self, parents):
		# Create a copy from one of the genes
		offspring = copy.deepcopy(parents[0])

		n = len(parents) - 1
		for i, chromosome in enumerate(offspring.genotype):
			chromosome = parents[randint(0, n)][i]

			offspring.mutate()
			offspring.decode()

			return offspring

	def select(self):
		# Keep the population size constant
		self.population = self.population[-self.popsize:]

	def roulette(self):
		# TODO: Implement Roulette selection
		pass

	def fittest(self):
		return self.population[-1]

	def evolve(self):
		return True if self.error > self.target_error else False
