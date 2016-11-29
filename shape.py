import os
import sys
import random
import signal
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from neuralnet import *
from nnmath import *
from genetics import GeneticAlgorithm

class GAKill(Exception):
	def __init__(self, message):
		self.message = message

def main(argv):
	# Set Numpy warning level
	np.seterr(over='ignore')

	# Define target shapes
	targets = np.array(['rectangle', 'circle'])

	if argv[1] == 'train':
		# Check the input arguments
		if len(argv) < 3:
		    print "Usage: python shape.py train <error>"
		    sys.exit()

		# Load the training data
		training_data = []
		for (dirpath, dirnames, filenames) in os.walk('./training_data/'):
			for dirname in dirnames:
				for f in os.listdir(dirpath + dirname):
					try:
						img = np.ravel(misc.imread(dirpath + dirname + '/' + f, flatten=True))
						training_data.append((dirname, img))
					except:
						pass

		# Shuffle for more randomness
		random.shuffle(training_data)

		# Create a GA of neural nets
		img_len = len(training_data[0][1])
		ga = GeneticAlgorithm(error = float(argv[2]),
								mutation_rate = 0.01,
								data = training_data,
								targets = targets,
								obj = NeuralNet,
								args = ([img_len, 50, 25, 2], logsig))

		# Create the 1st generation
		print "Creating population..."
		ga.populate(200)

		print "Initiating GA heuristic approach..."

		# Start evolution
		epoch = 0
		errors = []
		while ga.evolve():
			try:
				ga.evaluate()
				ga.crossover()

				# Store error and measure time
				epoch += 1
				errors.append(ga.error)
				print "error: " + str(ga.error)
			except GAKill as e:
				print e.message
				break

		print "error: " + str(ga.error)
		print "--------------------------------------------------------------\n"

		# Write the weights to file
		nn = ga.fittest()
		nn.save("neuralnet.pkt")

		x = range(epoch)
		y = errors

		# Plot error over time
		fig = plt.figure()
		plt.plot(x, y)
		plt.show()

		print "Done!"

	elif argv[1] == "predict":
		# Check the arguments
		if len(argv) < 3:
		    print "Usage: python shape.py test <image>"
		    sys.exit()

		# Read the test image
		img = np.ravel(misc.imread(argv[2], flatten=True))

		# Build the neural net from file
		nn = NeuralNet(([], logsig), build=False)
		nn.load("neuralnet.pkt")

		output = nn.feed_forward(img)

		# Determine the result
		result = targets[np.around(output).astype(np.bool)]

		print (output, result)

	else:
		print "ERROR: Unknown command " + argv[1]


def signal_handler(signal, frame):
	raise(GAKill("\nAborting Search..."))

if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)
	main(sys.argv)
