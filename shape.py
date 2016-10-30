import sys
from genetics import GeneticAlgorithm
from neuralnet import NeuralNet

def main(argv):
	if (argv[2] is "train"):
		targets =
		training_data =

		generations = 20
		mutation_rate = 0.1
		ga = GeneticAlgorithm(generations, mutation_rate, NeuralNet, targets)
		ga.population(20)

		i = 0
		while ga.evolve():
			ga.evaluate(training_data[i])
			ga.select()
			ga.breed()
			++i

		# Switch to a gradient search
		nn = ga.fittest()
		for data in enumerate(training_data, start=i):
			nn.backpropagate(data)

		# TODO: Store the weights

	else if argv[2] is "predict":
		test_data =
		# TODO
	else:
		print "Unknown command " + argv[1]

if __name__ == "__main__":
	main(str(sys.argv))
