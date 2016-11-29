import numpy as np
from random import randint

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
                self.genotype[idx] += 2 * np.random.random_sample(1) - 1 # Add a small -/+ number

	def read_genotype(self, delta):
		chunk = self.genotype[self.cursor:self.cursor + delta]
		self.cursor += delta
		return chunk


class GeneticAlgorithm:
	def __init__(self, error, mutation_rate, data, targets, obj, args):
		"""
		This contructor takes multiple parameters as well as the constructor
		for the population and an n-tuple for the arguments of the contructor.
		It is assumed the contructor knows how to decompose this.
		"""
		self.obj = obj
		self.args = args
		self.mutation_rate = mutation_rate
		self.target_error = error
		self.training_data = data
		self.targets = targets
		self.popsize = 0
		self.error = 1

	def populate(self, size):
		# Use the object constructor to create the population
		self.population = np.array([self.obj(self.args) for _ in range(size)])
		self.popsize = size

	def singleton(self):
		return self.obj(self.args, build=False)

	def evaluate(self):
		for gene in self.population:
			# Reset the score
			gene.score = 0
			for data in self.training_data:
				(tags, input_vector) = data
				output = gene.evaluate(input_vector)
				#print output

				# Determine the desired output
				target = map(lambda x: int(x in tags), self.targets)

				# Calculate how well it performed
				eval_vector = (target == np.around(output)).astype(np.int)

				# Build the scoring vector
				scorer = [0] * len(output)
				for i, value in enumerate(target):
					if value == 0:
						scorer[i] = 1 - output[i]
					else:
						scorer[i] = output[i]

				gene.score += np.dot(scorer, eval_vector)

			gene.fitness = gene.score/(len(eval_vector) * len(self.training_data))

		self.population = sorted(self.population, key=lambda gene: gene.fitness)
		self.error = 1 - self.fittest().fitness  # Set the error

	def crossover(self):
		offsprings = []

		# Select parents based on roulette selection
		for _ in range(self.popsize):
			offsprings.append(self.breed(self.roulette(2)))

		self.population = np.array(offsprings)

	def breed(self, parents):
		# Make a new gene and don't update global population size
		offspring = self.singleton()

		# Determine points of cut
		length = parents[0].genotype.size - 1
		cuts = [randint(0, length/2), randint(length/2, length)]
		# Perform 2-point crossover
		offspring.genotype = np.concatenate((parents[0].genotype[:cuts[0]], parents[1].genotype[cuts[0]:cuts[1]], parents[0].genotype[cuts[1]:]))
		offspring.mutate(self.mutation_rate)
		offspring.decode()

		return offspring

	def roulette(self, n):
		# Gather the fitnesses from all the gene
		fitnesses = map(lambda x: x.fitness, self.population)
		fitnesses /= np.sum(fitnesses) # Normalise

		return np.random.choice(self.population, n, p=fitnesses)

	def fittest(self):
		return self.population[-1]

	def evolve(self):
		return True if self.error > self.target_error else False
