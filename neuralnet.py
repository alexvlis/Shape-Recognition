import cPickle
import numpy as np
from genetics import Gene
from nnmath import *

class NeuralNet(Gene):
	weights = []
	biases = []
	error = 0
	alpha = 1
	n = 0

	def __init__(self, args, build=True):
		(self.skeleton, self.sigmoid, self.alpha) = args
		if build:
			self.build(self.skeleton)
			self.encode()

	def build(self, skeleton):
		for i, width in enumerate(skeleton[1:], start=1):
			weights = 5 * (2 * np.random.sample((width, skeleton[i-1])) - 1)
			biases = 2 * np.random.sample(width) - 1
			self.weights.append(weights)
			self.biases.append(biases)

		self.n = len(self.weights) + 1

	def feed_forward(self, activation):
		zs = []
		activations = [activation]
		z = activation
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = self.sigmoid(z)
			activations.append(activation)

		return (activations, zs)

	def backpropagate(self, activation, target):
		activations, zs = self.feed_forward(activation)

		# Determine the output deltas using the derivative of the sigmoid
		delta = sig_prime(zs[-1]) * (activations[-1] - target)

		# Adjust the weights and biases of the output layer
		self.weights[-1] -= self.alpha * delta[:, np.newaxis] * activations[-2][np.newaxis, :]
 		self.biases[-1] -= self.alpha * delta

		# Propagate error to the hidden layers
		for i in xrange(2, self.n):
			# Calculate the delta for this layer
			delta = np.dot(self.weights[-i+1].T, delta) * sig_prime(zs[-i])
			# Adjust the weights of this layer
			self.weights[-i] -= self.alpha * delta[:, np.newaxis] * activations[-i-1][np.newaxis, :]
			self.biases[-i] -= self.alpha * delta

		return square_error(target, activations[-1])

	def load(self, filename):
		'''This method sets the parameters of the neural net from file.'''
		with open(filename, "r") as f:
			self.weights, self.biases = cPickle.loads(f.read())

	def save(self, filename):
		'''This method saves the parameters of the neural net to file.'''
		with open(filename, "w+") as f:
			f.write(cPickle.dumps((self.weights,self.biases)))


	'''*********************** Overload Gene methods ************************'''

	def encode(self):
		'''Encode the network parameters into a series of real values'''
		genotype = np.array([]) # Initialise a new genotype
		for w, b in zip(self.weights, self.biases):
			genotype = np.concatenate((genotype, w.flatten(), b))

		self.genotype = genotype

	def decode(self):
		'''Decode genotype into layers of weights and biases'''
		for i, width in enumerate(self.skeleton[1:], start=1):
			d = (self.skeleton[i-1] + 1) * width
			# Read the weights for layer and reshape them to 2D
			weights = self.read_genotype(d).reshape(width, self.skeleton[i-1])
			biases = self.read_genotype(width)

			self.weights.append(weights)
			self.biases.append(biases)

		self.cursor = 0 # Reset the cursor

	def evaluate(self, inputs, targets, epochs):
		for i in range(epochs):
			for tag, img in inputs:
				target = map(lambda x: int(x in tag), targets)
				self.error += self.backpropagate(img, target)

			print "Epoch: " + str(i) + " error: " + str(self.error/len(inputs))
			self.error = 0 # Reset the error
