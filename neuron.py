import numpy as np

class Neuron:
	def __init__(self, size, beta, activation):
		self.w = np.random.random_sample(size)
		self.b = np.random.random_sample()
		self.beta = beta
		self.activation = activation

	def activate(self, inputs):
		return np.sum(self.activation(np.multiply(self.w, inputs) + self.b))

	def encode(self):
		return np.concatenate(self.w, self.b)
