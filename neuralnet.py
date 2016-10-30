from neuron import Neuron
from genetics import Gene
from nnmath import *

class NeuralNet(Gene):
	def __init__(self, targets):
		self.layers = []
		self.fitness = 0
		self.targets = targets
		self.build(np.array([25, 4, 2, 3]), logsig)
		# TODO: Define output layer

	def build(self, skeleton, activation):
		for i, width in enumerate(skeleton, start=1):
			layer = [Neuron(size=skeleton[i-1], beta=0.5, activation=activation) for i in range(width)]

			self.layers.append(layer)

	def output_layer(self, funcs):
		for neuron, func in zip(self.layers[-1], funcs):
			neuron.set_activation(func)

	def feed_forward(self, input_vector):
		for layer in self.layers:
			outputs = []
			for i, neuron in enumerate(layer):
				outputs[i] = neuron.compute(input_vector)

			input_vector = outputs

		return outputs

	def backpropagate(self, data):
		output = self.feed_forward(input_vector)
		# TODO: Finish implementation


	'''********************** Overload gene functions ***********************'''
	def mutate(self):
		pass

	def breed(self):
		pass

	def encode(self):
		pass
