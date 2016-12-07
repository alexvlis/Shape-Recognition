import numpy as np

tansig = lambda n: 2 / (1 + np.exp(-2 * n)) - 1

sigmoid = lambda n: 1 / (1 + np.exp(-n))

hardlim = lambda n: 1 if n >= 0 else 0

purelin = lambda n: n

relu = lambda n: np.fmax(0, n)

square_error = lambda x, y: np.sum(0.5 * (x - y)**2)

sig_prime = lambda z: sigmoid(z) * (1 - sigmoid(z))

relu_prime = lambda z: relu(z) * (1 - relu(z))

softmax = lambda n: np.exp(n)/np.sum(np.exp(n))

softmax_prime = lambda n: softmax(n) * (1 - softmax(n))

cross_entropy = lambda x, y: -np.dot(x, np.log(y))
