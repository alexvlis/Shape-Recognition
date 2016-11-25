import os
import sys
import random
from scipy import misc
from neuralnet import *
from genetics import GeneticAlgorithm

def main(argv):
	# Set Numpy warning level
	np.seterr(over='ignore')

	# Define target shapes
	targets = np.array(['rectangle', 'triangle', 'circle'])

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
		ga = GeneticAlgorithm(error, mutation_rate, NeuralNet, training_data, targets)
		ga.population(100)

		print "Initiating GA heuristic approach..."

		while ga.evolve():
			ga.evaluate()
			ga.select()
			ga.breed()

			print "error: " + str(ga.error)
			print "\n--------------------------------------------------------\n"

		# Write the weights to file
		nn = ga.fittest()
		with open("weights.txt", "w+") as f:
		    for layer in nn.layers:
		        for neuron in layer:
					for weight in neuron.w:
					    f.write(str(weight) + ";")
					f.write(str(neuron.b) + "\n")

		print "Done!"

	elif argv[1] == "test":
		if len(argv) < 3:
		    print "Usage: python shape.py test <image>"
		    sys.exit()

		nn = NeuralNet()
		with open("weights.txt", "r") as f:
			for layer in nn.layers:
				for neuron in layer:
					line = f.readline().split(";")
					neuron.w = np.array(line[0:-1]).astype(np.float)
					neuron.b = float(line[-1])

		img = np.ravel(misc.imread(argv[2], flatten=True))

		output = nn.feed_forward(img)
		result = targets[np.around(output).astype(np.bool)]
		print result

	else:
		print "ERROR: Unknown command " + argv[1]

if __name__ == "__main__":
	main(sys.argv)
