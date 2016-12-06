import struct
import numpy as np

tansig = lambda n: 2 / (1 + np.exp(-2 * n)) - 1

logsig = lambda n: 1 / (1 + np.exp(-n))

hardlim = lambda n: 1 if n >= 0 else 0

purelin = lambda n: n

square_error = lambda x, y: np.sum(0.5 * (x - y)**2)

sig_prime = lambda z: logsig(z) * (1 - logsig(z)) 
