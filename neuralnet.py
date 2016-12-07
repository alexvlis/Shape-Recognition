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
		(self.skeleton, self.alpha) = args
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
		# Propagate through hidden layers
		for w, b in zip(self.weights[:-1], self.biases[:-1]):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		# Use softmax for output layer
		z = np.dot(self.weights[-1], activation) + self.biases[-1]
		zs.append(z)
		activations.append(sigmoid(z))
		return (activations, zs)

	def backpropagate(self, activation, target):
		# Initialise the deltas
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activations, zs = self.feed_forward(activation)
		self.error += square_error(target, activations[-1])

		# Determine the output deltas using the derivative of the sigmoid
		delta = softmax_prime(zs[-1]) * (activations[-1] - target)

		nabla_w[-1] = delta[:, None] * activations[-2][None, :]
		nabla_b[-1] = delta

		# Propagate error to the hidden layers
		for i in xrange(2, self.n):
			# Calculate the delta for this layer
			delta = np.dot(self.weights[-i+1].T, delta) * sig_prime(zs[-i])
			#
			nabla_w[-i] = delta[:, None] * activations[-i-1][None, :]
			nabla_b[-i] = delta

		return (nabla_w, nabla_b)

	def gradient_descent(self, inputs, targets, epochs):
		m = len(inputs)

		for i in range(epochs):
			nabla_b = [np.zeros(b.shape) for b in self.biases]
			nabla_w = [np.zeros(w.shape) for w in self.weights]
			self.error = 0 # Reset the error

			for tag, img in inputs:
				target = map(lambda x: int(x in tag), targets)
				delta_nabla_w, delta_nabla_b = self.backpropagate(img, target)

				for j in range(len(nabla_w)):
					nabla_w[j] += delta_nabla_w[j]
					nabla_b[j] += delta_nabla_b[j]

			# Update the weights and biases
			self.weights = [w-(self.alpha/m)*nw for w, nw in zip(self.weights, nabla_w)]
			self.biases = [b-(self.alpha/m)*nb for b, nb in zip(self.biases, nabla_b)]

			self.error /= m # Normalize the error
			print "Epoch: " + str(i) + " error: " + str(self.error)

			self.fitness = 1 - self.error # Set the fitness

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
		self.weights = []
		self.biases = []
		for i, width in enumerate(self.skeleton[1:], start=1):
			d = (self.skeleton[i-1] + 1) * width
			# Read the weights for layer and reshape them to 2D
			weights = self.read_genotype(d).reshape(width, self.skeleton[i-1])
			biases = self.read_genotype(width)

			self.weights.append(weights)
			self.biases.append(biases)

		self.cursor = 0 # Reset the cursor

	# Wrapper around the gradient descent method
	def evaluate(self, inputs, targets, epochs):
		self.gradient_descent(inputs, targets, epochs)
