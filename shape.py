import sys
import os
from genetics import GeneticAlgorithm
from neuralnet import *
from scipy import misc

def main(argv):
	if (str(argv[1]) == 'train'):
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

		# Define target shapes
		targets = ['rectangle', 'triangle', 'circle']

		mutation_rate = 0.1
		error = 0.3
		ga = GeneticAlgorithm(error, mutation_rate, NeuralNet, training_data, targets)
		ga.population(100)

		print "Initiating GA heuristic approach..."

		while ga.evolve():
			ga.evaluate()
			ga.select()
			ga.breed()

			print "error: " + str(1 - ga.fittest().fitness)
			print "\n--------------------------------------------------------\n"

		# Switch to a gradient search
		# print "Attempting gradient search..."

		# nn = ga.fittest()
		# for idx, data in enumerate(training_data, start=i):
		# 	try:
		# 		nn.backpropagate(data)
		# 	except ErrorMinimized as e:
		# 		print e.message
		# 		break

		print "Done!"
		# TODO: Store the weights

	elif str(argv[1]) == "test":
		test_data = []
		# TODO
	else:
		print "ERROR: Unknown command " + argv[1]

if __name__ == "__main__":
	main(sys.argv)
