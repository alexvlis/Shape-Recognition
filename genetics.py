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
		"""
		This contructor takes multiple parameters as well as the constructor
		for the population and an n-tuple for the arguments of the contructor.
		It is assumed the contructor know how to decompose this.
		"""
		self.obj = obj
		self.mutation_rate = mutation_rate
		self.target_error = error
		self.training_data = data
		self.targets = targets
		self.popsize = 0
		self.error = 1

	def population(self, size):
		# Use the object constructor to create the population
		self.population = np.array([self.obj(args) for _ in range(size)])
		self.popsize = size

	def evaluate(self):
		for gene in self.population:
			# Reset the score
			gene.score = 0
			for data in self.training_data:
				(tags, input_vector) = data
				output = gene.evaluate(input_vector)

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
		for gene in self.population:
			# Breed with random gene from the population
			offspring = gene.breed(self.population[randint(0, self.popsize-2)], self.mutation_rate)

			offsprings.append(offspring)

		self.population = self.population + offsprings

	def crossover(self):
		# Select parents based on roulette selection
		for _ in range(self.popsize):
			parents = self.roulette(2)
			offspring = self.breed(parents)

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

	def roulette(self, n):
		# Gather the fitnesses from all the gene
		fitnesses = map(lambda x: x.fitness, self.population)
		fitnesses /= np.sum(fitnesses) # Normalise

		return np.random.choice(self.population, n, p=fitnesses)

	def fittest(self):
		return self.population[-1]

	def evolve(self):
		return True if self.error > self.target_error else False
