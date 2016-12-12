import cPickle
import numpy as np
import matplotlib.pyplot as plt
from genetics import Gene
from nnmath import *

class NeuralNet(Gene):
	errors = []
	test_accuracies = []
	train_accuracies = []
	alpha_max = 0.8
	alpha_min = 0.01
	decay_speed = 1000

	def __init__(self, args, build=True):
		self.biases = []
		self.weights = []
		self.skeleton = args
		if build:
			self.build(self.skeleton)
			self.encode()

	def build(self, skeleton):
		for i, width in enumerate(skeleton[1:], start=1):
			weights = (2 * np.random.sample((width, skeleton[i-1])) - 1)
			biases = (2 * np.random.sample(width) - 1)
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
		activations.append(softmax(z))
		return (activations, zs)

	def backpropagate(self, activation, target):
		# Initialise the deltas
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		activations, zs = self.feed_forward(activation)
		self.errors[-1] += square_error(target, activations[-1])
		if np.argmax(target) == np.argmax(activations[-1]):
			self.train_accuracies[-1] += 1

		# Determine the output deltas using the derivative of the sigmoid
		delta = softmax_prime(zs[-1]) * (activations[-1] - target)

		nabla_w[-1] = delta[:, None] * activations[-2][None, :]
		nabla_b[-1] = delta

		# Propagate error to the hidden layers
		for i in xrange(2, self.n):
			# Calculate the delta for this layer
			delta = np.dot(self.weights[-i+1].T, delta) * sig_prime(zs[-i])

			nabla_w[-i] = delta[:, None] * activations[-i-1][None, :]
			nabla_b[-i] = delta

		return (nabla_w, nabla_b)

	def gradient_descent(self, training_data, test_data, targets, epochs, vis=False):
		if vis:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			error_line, = ax.plot(x, y, 'b-')
			test_accuracy_line, = ax.plot(x, y, 'r-')
			test_accuracy_line, = ax.plot(x, y, 'g-')

		m = len(training_data)

		for i in range(epochs):
			nabla_b = [np.zeros(b.shape) for b in self.biases]
			nabla_w = [np.zeros(w.shape) for w in self.weights]
			self.errors.append(0) # Reset the error
			self.train_accuracies.append(0)

			for tag, img in training_data:
				target = map(lambda x: int(x in tag), targets)
				delta_nabla_w, delta_nabla_b = self.backpropagate(img, target)

				for j in range(self.n - 1):
					nabla_w[j] += delta_nabla_w[j]
					nabla_b[j] += delta_nabla_b[j]

			# Update the weights and biases
			self.weights = [w-(self.learning_rate(i)/m)*nw for w, nw in zip(self.weights, nabla_w)]

			self.biases = [b-(self.learning_rate(i)/m)*nb for b, nb in zip(self.biases, nabla_b)]

			self.test_accuracies.append(0)
			for tag, img in test_data:
				target = map(lambda x: int(x in tag), targets)
				activations, zs = self.feed_forward(img)

				if np.argmax(target) == np.argmax(activations[-1]):
					self.test_accuracies[-1] += 1

			self.errors[-1] /= m # Normalize the error
			self.test_accuracies[-1] /= float(len(test_data))
			self.train_accuracies[-1] /= float(m)
			print "Epoch: " + str(i) + " error: " + str(self.errors[-1]) + " accuracy: " + str(self.test_accuracies[-1]) + " train_accuracy: " + str(self.train_accuracies[-1])

			# Update the visualization
			if vis:
				# Update the time axis
				error_line.set_ydata(range(i))
				train_accuracy_line.set_ydata(range(i))
				test_accuracy_line.set_ydata(range(i))

				error_line.set_xdata(self.errors)
				train_accuracy_line.set_xdata(self.train_accuracies)
				test_accuracy_line.set_xdata(self.test_accuracies)
				fig.canvas.draw() # Plot everything

	def validate(self, targets, test_data):
		accuracy = 0.0
		for tag, img in test_data:
			target = map(lambda x: int(x in tag), targets)
			activations, zs = self.feed_forward(img)

			if np.argmax(target) == np.argmax(activations[-1]):
				accuracy += 1

		return accuracy/len(test_data)

	def learning_rate(self, i):
		return self.alpha_min + (self.alpha_max - self.alpha_min) * np.exp(-i/self.decay_speed)

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
			d = (self.skeleton[i-1]) * width
			# Read the weights for layer and reshape them to 2D
			weights = self.read_genotype(d).reshape(width, self.skeleton[i-1])
			biases = self.read_genotype(width)

			self.weights.append(weights)
			self.biases.append(biases)

		self.cursor = 0 # Reset the cursor
		self.n = len(self.weights) + 1

	# Wrapper around the gradient descent method
	def evaluate(self, training_data, targets):
		score = 0
		for tag, img in training_data:
			target = map(lambda x: int(x in tag), targets)
			activations, zs = self.feed_forward(img)

			eval_vec = (np.round(activations[-1]) == target).astype(np.int)
			score += np.dot(activations[-1], eval_vec)
			eval_vec = (np.round(activations[-1]) != target).astype(np.int)
			score -= np.dot(activations[-1], eval_vec)

		if score < 0:
			score = 0

		self.fitness = score/len(training_data)
