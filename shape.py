from neuralnet import *
from nnmath import *

def main():
	# Create a neural net
	nn = NeuralNet()

	# Build the network
	nn.build(np.array([4, 2, 1]), logsig)
	# TODO: Define output layer

	nn.train()
	nn.test()

if __name__ == "__main__":
	main()
