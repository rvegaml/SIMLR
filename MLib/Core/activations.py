'''
File: activations.py
Author: Roberto Vega
Description:
	This file contains common activations functions used in my functions.
'''
import numpy as np

def sigmoid(x):

	y = 1 / (1 + np.exp(-x))

	return y

def identity(x):
	return x
