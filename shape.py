from neuralnet import *

net = NeuralNet()

tansig = lambda n: 2 / (1 + np.exp(-2 * n)) - 1
logsig = lambda n: 1 / (1 + np.exp(-n))

net.build(np.array([4, 2, 1]), logsig)
# TODO: Define output layer

net.train()
net.test()
