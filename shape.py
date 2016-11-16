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

		mutation_rate = 0.8
		error = 0.5
		ga = GeneticAlgorithm(error, mutation_rate, NeuralNet, training_data, targets)
		ga.population(20)

		print "Initiating GA heuristic approach..."

		while ga.evolve():
			ga.evaluate()
			ga.select()
			ga.breed()

			print "error: " + str(1 - ga.fittest().fitness)
			print "target error: " + str(error)

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

	elif str(argv[1]) == "predict":
		test_data = 0
		# TODO
	else:
		print "ERROR: Unknown command " + argv[1]

if __name__ == "__main__":
	main(sys.argv)
