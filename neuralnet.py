from neuron import Neuron
from genetics import Gene
from nnmath import *

class NeuralNet(Gene):
	def __init__(self, targets):
		self.layers = []
		self.targets = targets
		self.build(np.array([25, 4, 2, 1]), logsig)
		# TODO: Define output layer

	def build(self, skeleton, activation):
		for i, width in enumerate(skeleton, start=1):
			layer = [Neuron(size=skeleton[i-1], beta=0.5, activation=activation) for i in range(width)]

			self.layers.append(layer)
		self.encode()

	def feed_forward(self, input_vector):
		for layer in self.layers:
			outputs = []
			for neuron in layer:
				outputs.append(neuron.activate(input_vector))

			input_vector = outputs

		return outputs

	def backpropagate(self, data):
		output = self.feed_forward(input_vector)
		# TODO: Finish implementation


	'''*********************** Overload gene methods ************************'''
	def mutate(self):
		# TODO: Mutate its parameters

	def encode(self):
		for layer in self.layers:
			for neuron in layer:
				self.genome.append(neuron.encode())

	def evaluate(self, input_vector):
		return self.feed_forward(input_vector)

	def breed(self, nn):
		offspring = copy.deepcopy(nn)
		for code, offspring_code in zip(self.genome, offspring.genome):
			# TODO: combine the codes and put it in the offspring code
			# TODO: Also change the actual weights of the offspring neurons
			offspring_code = code + offspring_code
