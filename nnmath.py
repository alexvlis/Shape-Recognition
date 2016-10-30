import numpy as np

tansig = lambda n: 2 / (1 + np.exp(-2 * n)) - 1

logsig = lambda n: 1 / (1 + np.exp(-n))

hardlim = lambda n: 1 if n >= 0 else 0

purelin = lambda n: n

def euclidean(x, y):
	return np.sqrt(np.sum((x-y)**2))
