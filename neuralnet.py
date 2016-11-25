import numpy as np
import copy
from genetics import Gene
from nnmath import *

class ErrorMinimized(Exception):
	def __init__(self, message):
		self.message = message


class Neuron:
	def __init__(self, size, activation):
		self.w = 6 * (2 * np.random.random_sample(size) - 1)
		self.b = 2 * np.random.random_sample() - 1
		self.activation = activation

	def activate(self, inputs):
		return self.activation(np.sum((np.multiply(self.w, inputs) + self.b)))

	def mutate(self, rate):
		self.w += rate * np.random.random_sample() * round(2 * np.random.random_sample() - 1)
		self.b += rate * np.random.random_sample() * round(2 * np.random.random_sample() - 1)


class NeuralNet(Gene):
	def __init__(self, input_len):
		self.layers = []
		self.build([input_len, 4, 4, 3], logsig)

	def build(self, genome, activation):
		for i, width in enumerate(genome[1:], start=1):
			layer = [Neuron(size=genome[i-1], activation=activation) for j in range(width)]
			self.layers.append(layer)

	def feed_forward(self, input_vector):
		for layer in self.layers:
			outputs = []
			for neuron in layer:
				outputs = np.append(outputs, neuron.activate(input_vector))

			input_vector = outputs

		return outputs

	def backpropagate(self, data):
		output = self.feed_forward(input_vector)
		# TODO: Finish implementation
		raise(ErrorMinimized("Error Minimized!"))


	'''*********************** Overload gene methods ************************'''

	def mutate(self, rate):
		for layer in self.layers:
			for neuron in layer:
				neuron.mutate(rate)

	def evaluate(self, input_vector):
		return self.feed_forward(input_vector)

	def breed(self, parent, mutation_rate):
		offspring = copy.deepcopy(parent)
		fitness_sum = self.fitness + parent.fitness
		if fitness_sum == 0:
			weight1 = 0.5
			weight2 = 0.5
		else:
			weight1 = self.fitness/fitness_sum
			weight2 = parent.fitness/fitness_sum

		for layer, off_layer in zip(self.layers, offspring.layers):
			for neuron, off_neuron in zip(layer, off_layer):
				off_neuron.w = weight1 * neuron.w + weight2 * off_neuron.w
				off_neuron.b = weight1 * neuron.b + weight2 * off_neuron.b

				# Mutate the offspring and the parent neurons
				neuron.mutate(mutation_rate)
				off_neuron.mutate(mutation_rate)

		return offspring
