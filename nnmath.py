import numpy as np
import struct

tansig = lambda n: 2 / (1 + np.exp(-2 * n)) - 1

logsig = lambda n: 1 / (1 + np.exp(-n))

hardlim = lambda n: 1 if n >= 0 else 0

purelin = lambda n: n

def float_to_bin(x):
	if x == 0:
		return "0" * 64

	w, sign = (float.hex(x), 0) if x > 0 else (float.hex(x)[1:], 1)
	mantissa, exp = int(w[4:17], 16), int(w[18:])

	return "{}{:011b}{:052b}".format(sign, exp + 1023, mantissa)

def bin_to_float(x):
	x = '0b' + x
	q = int(x, 0)
	b8 = struct.pack('Q', q)

	return struct.unpack('d', b8)[0]
