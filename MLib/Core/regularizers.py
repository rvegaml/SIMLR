'''
File: losses.py
Author: Roberto Vega
Description:
	This file contains common regularizers used in my functions.
'''
import numpy as np

class l2_reg():
	'''
	This class implements a simple l2 regularizer
	'''
	def __init__(self, alpha=0):
		'''
		Description:
			This function initializes the attributes of the l2 regularizer
		Args:
			alpha: Constant value that indicates the weight of the regularer.
		Returns:
			None
		'''

		self.alpha = alpha

	def compute_loss(self, theta):
		'''
		Description:
			This function computes alpha*sum(w^T w)
		Args:
			theta: Numpy array that contains the weights to be regularized
		Returns:
			reg_loss: Value of alpha*sum(w^T w)
		'''

		# Compute the square of every value
		sqr = np.square(theta)

		return self.alpha * np.sum(sqr)

	def compute_gradient(self, theta):
		'''
		Description:
			This function computes the gradient of the regularizer
		Args:
			theta: Numpy array that contains the weights to be regularized
		Returns:
			reg_loss: Value of the gradient of alpha*sum(w^T w)
		'''

		gradient = 2*self.alpha*theta
		
		return gradient

class l2_initial_weights_reg():
	'''
	This class implements a simple l2 regularizer on the difference between
	the initial weights and the current weights.
	'''
	def __init__(self, alpha=0, initial_weights=None):
		'''
		Description:
			This function initializes the attributes of the l2 regularizer
		Args:
			alpha: Constant value that indicates the weight of the regularer.
		Returns:
			None
		'''

		self.alpha = alpha
		self.init_w = initial_weights

	def compute_loss(self, theta):
		'''
		Description:
			This function computes alpha*sum(w^T w)
		Args:
			theta: Numpy array that contains the weights to be regularized
		Returns:
			reg_loss: Value of alpha*sum(w^T w)
		'''

		# Compute the square of every value
		sqr = np.square(theta-self.init_w)

		return self.alpha * np.sum(sqr)

	def compute_gradient(self, theta):
		'''
		Description:
			This function computes the gradient of the regularizer
		Args:
			theta: Numpy array that contains the weights to be regularized
		Returns:
			reg_loss: Value of the gradient of alpha*sum(w^T w)
		'''

		gradient = 2*self.alpha*(theta-self.init_w)
		
		return gradient