import cPickle
import numpy as np
from genetics import Gene

class NeuralNet(Gene):
	def __init__(self, args, build=True):
		(skeleton, activation) = args
		self.layers = []
		self.activation = activation
		self.skeleton = skeleton
		self.cursor = 0

		if build:
			self.build(skeleton)
			self.encode()

	def build(self, skeleton):
		for i, width in enumerate(skeleton[1:], start=1):
			weights = 5 * (2 * np.random.sample((skeleton[i-1], width)) - 1)
			biases = 5 * (2 * np.random.sample(width) - 1)
			self.layers.append((weights, biases))

	def feed_forward(self, input_vec):
		for (weights, biases) in self.layers:
			input_vec = self.activation(np.dot(input_vec, weights) + biases)

		return input_vec

	def load(self, filename):
		'''This method sets the parameters of the neural net from file.'''
		with open(filename, "r") as f:
			self.layers = cPickle.loads(f.read())

	def save(self, filename):
		with open(filename, "w+") as f:
			f.write(cPickle.dumps(self.layers))


	'''*********************** Overload gene methods ************************'''

	def encode(self):
		genotype = np.array([]) # Initialise a new genotype
		for weights, biases in self.layers:
			genotype = np.concatenate((genotype, weights.flatten(), biases))

		self.genotype = genotype

	def decode(self):
		for i, width in enumerate(self.skeleton[1:], start=1):
			d = skeleton[i-1] * width
			weights = self.read_genotype(d).reshape(self.skeleton[i-1], width)
			biases = self.read_genotype(width)

			self.layers.append((weight, biases))

		self.cursor = 0 # Reset the cursor

	def evaluate(self, input_vector):
		return self.feed_forward(input_vector)

	def read_genotype(self, delta):
		chunk = self.genotype[self.cursor:self.cursor + delta]
		self.cursor += delta
		return chunk
