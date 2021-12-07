'''
Common kernels
'''
import numpy as np


def linear_kernel(x, y):
	'''
	Linear kernel is simply he dot product of the vectors x and y.
	'''
	return np.dot(x, y)

def gaussian_kernel(x, y, sigma=0.5):
	'''
	'''
	numerator = np.dot(x-y, x-y)
	denominator = 2*sigma*sigma

	return np.exp(-numerator / denominator)

