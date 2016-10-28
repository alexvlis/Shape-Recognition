import numpy as np

class Neuron:
	def __init__(self, size, beta, activation, bias):
		self.w = np.random.random_sample(size)
		self.b = bias
		self.beta = beta
		self.activation = activation

	def compute(self, inputs):
		return np.sum(self.activation(np.multiply(self.w, inputs) + self.b))

	def activate(self, inputs, t):
		output = self.compute(inputs)
		self.w += self.beta * (output - t) * inputs
		return output
