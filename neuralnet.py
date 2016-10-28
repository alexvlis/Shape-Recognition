from neuron import *

class NeuralNet:
	def __init__(self):
		self.layers = null

	def build(self, skeleton, activation):
		for width in skeleton:
			layer = [Neuron(size=2, beta=0.5, bias=0, activation=activation) for i in range(width)]

			self.layers.append(layer)

	def output_layer(self, decisions):
		pass

	def train(self, training_data):
		for inputs in training_data:
			outputs = []
			for layer in self.layers:
				for i, neuron in enumerate(layer):
					outputs[i] = neuron.activate(inputs, ???)

				inputs = outputs
				outputs = []

	def test(self):
		pass

	def predict(self):
		pass
