import numpy as np
import copy
from genetics import Gene
from nnmath import *

class ErrorMinimized(Exception):
	def __init__(self, message):
		self.message = message


class Neuron:
	def __init__(self, size, activation):
		self.w = 4 * np.random.random_sample(size)
		self.b = np.random.random_sample()
		self.activation = activation

	def activate(self, inputs):
		return self.activation(np.sum((np.multiply(self.w, inputs) + self.b)))

	def encode(self):
		return np.append(self.w, self.b)

	def mutate(self, rate):
		self.w += rate * np.random.random_sample()
		self.b += rate * np.random.random_sample()


class NeuralNet(Gene):
	def __init__(self):
		self.layers = []
		self.build([2500, 5, 5, 4, 3], logsig)
		self.encode()

	def build(self, skeleton, activation):
		for i, width in enumerate(skeleton[1:], start=1):
			layer = [Neuron(size=skeleton[i-1], activation=activation) for j in range(width)]
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

	def encode(self):
		for layer in self.layers:
			for neuron in layer:
				self.genome.append(neuron.encode())

	def evaluate(self, input_vector):
		return self.feed_forward(input_vector)

	def breed(self, parent):
		offspring = copy.deepcopy(parent)
		for layer, off_layer in zip(self.layers, offspring.layers):
			for neuron, off_neuron in zip(layer, off_layer):
				off_neuron.w =
				off_neuron.b =

		return offspring
