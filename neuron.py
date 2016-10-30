import numpy as np

class Neuron:
	def __init__(self, size, beta, activation):
		self.w = np.random.random_sample(size)
		self.b = np.random.random_sample()
		self.beta = beta
		self.activation = activation

	def compute(self, inputs):
		return np.sum(self.activation(np.multiply(self.w, inputs) + self.b))

	def activate(self, inputs, t):
		output = self.compute(inputs)
		self.w += self.beta * (output - t) * inputs
		return output

	def set_activation(self, func):
		self.activation = func
