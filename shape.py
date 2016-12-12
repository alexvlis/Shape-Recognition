import os
import sys
import random
import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from neuralnet import *
from nnmath import *
from genetics import GeneticAlgorithm, GAKill

def read_data(path):
	data = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		for dirname in dirnames:
			for f in os.listdir(dirpath + dirname):
				try:
					img = np.ravel(misc.imread(dirpath + dirname + '/' + f, flatten=True))/255
					data.append((dirname, img))
				except:
					pass
	return data

def main(argv):
	# Set Numpy warning level
	np.seterr(over='ignore')

	# Define target shapes
	targets = np.array(['rectangle', 'circle', 'triangle'])

	if argv[1] == 'train':
		# Check the input arguments
		if len(argv) < 3:
		    print "Usage: python shape.py train <error>"
		    sys.exit()

		# Load the training data
		training_data = read_data('training_data/')
		test_data = read_data('test_data/')

		# Shuffle for more randomness
		random.shuffle(training_data)

		# Create a GA of neural nets
		img_len = len(training_data[0][1])
		ga = GeneticAlgorithm(epochs = int(argv[2]),
								mutation_rate = 0.01,
								data = training_data,
								targets = targets,
								obj = NeuralNet,
								args = [img_len, 20, 10, 3])

		# Create the 1st generation
		print "Creating population..."
		ga.populate(200)

		print "Initiating GA heuristic approach..."

		# Start evolution
		errors = []
		while ga.evolve():
			try:
				ga.evaluate()
				ga.crossover()
				ga.epoch += 1

				# Store error
				errors.append(ga.error)
				print "error: " + str(ga.error)
			except GAKill as e:
				print e.message
				break

		x = range(ga.epoch)
		y = errors

		# Plot error over time
		fig = plt.figure()
		plt.plot(x, y)
		plt.xlabel('Time (Epochs)')
		plt.ylabel('Error')
		plt.show()

		print "--------------------------------------------------------------\n"

		nn = ga.fittest()
		print "Initiating Gradient Descent optimization..."
		try:
			nn.gradient_descent(training_data, test_data, targets, int(argv[3]))
		except GAKill as e:
			print e.message

		nn.save("neuralnet.pkt")
		print "Done!"

	elif argv[1] == "validate":
		test_data = read_data('test_data/')
		
		nn = NeuralNet([], build=False)
		nn.load("neuralnet.pkt")

		accuracy = nn.validate(targets, test_data)
		print "Accuracy: " + str(accuracy)

	elif argv[1] == "predict":
		# Check the arguments
		if len(argv) < 3:
		    print "Usage: python shape.py test <image>"
		    sys.exit()

		# Read the test image
		img = np.ravel(misc.imread(argv[2], flatten=True))/255

		# Build the neural net from file
		nn = NeuralNet([], build=False)
		nn.load("neuralnet.pkt")

		# Predict
		activations, zs = nn.feed_forward(img)

		print targets[np.argmax(activations[-1])]

	else:
		print "ERROR: Unknown command " + argv[1]


def signal_handler(signal, frame):
	raise(GAKill("\nAborting Search..."))

if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)
	main(sys.argv)
