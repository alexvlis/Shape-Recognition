import os
import sys
import random
import signal
from scipy import misc
from neuralnet import *
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

		mutation_rate = 0.1
		error = float(argv[2])
		img_len = len(training_data[0][1])
		ga = GeneticAlgorithm(error = error,
								mutation_rate = mutation_rate,
								data = training_data,
								targets = targets,
								obj = NeuralNet,
								args = ([img_len, 200, 50, 4, 2], logsig))
		# Create the 1st generation
		ga.population(100)

		print "Initiating GA heuristic approach..."

		# Start evolution
		while ga.evolve():
			try:
				ga.evaluate()
				ga.select()
				ga.crossover()
				print "error: " + str(ga.error)
			except GAKill as e:
				print e.message
				break

		print "error: " + str(ga.error)
		print "--------------------------------------------------------------\n"

		# Write the weights to file
		nn = ga.fittest()
		nn.save("weights.txt")

		print "Done!"

	elif argv[1] == "test":
		if len(argv) < 3:
		    print "Usage: python shape.py test <image>"
		    sys.exit()

		img = np.ravel(misc.imread(argv[2], flatten=True))
		nn = NeuralNet(len(img))
		nn.load("weights.txt")

		output = nn.feed_forward(img)
		result = targets[np.around(output).astype(np.bool)]

		print (output, result)

	else:
		print "ERROR: Unknown command " + argv[1]


def signal_handler(signal, frame):
	raise(GAKill("\nAborting Search..."))

if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)
	main(sys.argv)
