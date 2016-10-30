from genetics import GeneticAlgorithm
from neuralnet import NeuralNet

def main():
	targets =
	training_data =

	ga = GeneticAlgorithm(NeuralNet, targets)
	ga.population(20)

	i = 0
	while ga.evolve():
		ga.evaluate(training_data[i])
		ga.select()
		ga.breed()
		++i

	# Switch to a gradient search
	nn = ga.best_gene()
	for data in enumerate(training_data, start=i):
		nn.backpropagate(data)

	# TODO: Store the weights

if __name__ == "__main__":
	main()
