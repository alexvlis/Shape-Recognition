import numpy as np
from genetics import Gene
from nnmath import *
import cStringIO

class ErrorMinimized(Exception):
	def __init__(self, message):
		self.message = message


class Neuron:
	def __init__(self, size, activation):
		self.w = 5 * (2 * np.random.random_sample(size) - 1)
		self.b = 1 * (2 * np.random.random_sample() - 1)
		self.activation = activation

	def activate(self, inputs):
		return self.activation(np.dot(self.w, inputs) + self.b)

	def mutate(self, rate):
		self.w += rate * np.random.random_sample() * round(2 * np.random.random_sample() - 1)
		self.b += rate * np.random.random_sample() * round(2 * np.random.random_sample() - 1)


class NeuralNet(Gene):
	layers = []

	def __init__(self, args):
		self.build(args[0], args[1])

	def build(self, skeleton, activation):
		for i, width in enumerate(skeleton[1:], start=1):
			layer = [Neuron(size=skeleton[i-1], activation=activation) for _ in range(width)]
			self.layers.append(layer)

	def feed_forward(self, input_vector):
		for layer in self.layers:
			outputs = []
			for neuron in layer:
				outputs = np.append(outputs, neuron.activate(input_vector))

			input_vector = outputs

		return outputs

	def load(self, filename):
		with open(filename, "r") as f:
			for layer in self.layers:
				for neuron in layer:
					line = f.readline().split(";")
					neuron.w = np.array(line[0:-1]).astype(np.float)
					neuron.b = float(line[-1])

	def save(self, filename):
		with open(filename, "w+") as f:
		    for layer in self.layers:
		        for neuron in layer:
					for weight in neuron.w:
					    f.write(str(weight) + ";")
					f.write(str(neuron.b) + "\n")


	'''*********************** Overload gene methods ************************'''

	def encode(self):
		for layer in self.layers:
			for neuron in layer:
				for weight in neuron.w:
					self.genotype += float_to_bin(weight)
				self.genotype += float_to_bin(neuron.b)

	def decode(self):
		genotype = cStringIO(self.genotype)
		for layer in self.layers:
			for neuron in layer:
				for weight in neuron.w:
					weight = bin_to_float(genotype.read(64))
				neuron.b = bin_to_float(genotype.read(64))

	def evaluate(self, input_vector):
		return self.feed_forward(input_vector)
