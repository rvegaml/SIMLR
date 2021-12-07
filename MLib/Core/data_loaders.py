'''
File: data_loaders.py
Author: Roberto Vega
Description:
	This file contains several data loaders
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import os


class DataSampler():
	'''
	This function generates samples to be fed to a neural network
	'''
	def __init__(self, batch_size=1):
		self.batch_size = batch_size
		self.index = -1
		# self.num_instances = X.shape[0]
		# self.order = np.random.permutation(self.num_instances)
		self.next_epoch = False

	def check_if_shuffle(self):
		'''
		This function checks if we need to shuffle the data. This condition
		happens when the index is greater than the number of samples
		'''

		if self.index >= self.num_instances:
			self.order = np.random.permutation(self.num_instances)
			self.index = 0
			self.next_epoch = True

	def get_data(self, X, y):
		'''
		This function returns the data that should be used for the current 
		batch.
		'''
		self.next_epoch = False

		if self.index == -1:
			self.index = 0
			self.num_instances = len(X)
			self.order = np.random.permutation(self.num_instances)

		# Get the indexes of the data to sample
		start_index = self.index
		end_index = start_index + self.batch_size

		if end_index > self.num_instances:
			end_index = self.num_instances

		batch_indexes = self.order[start_index:end_index]

		X_batch = list()
		y_batch = list()

		for element in batch_indexes:
			X_batch.append(X[element])
			y_batch.append(y[element])

		X_batch = np.array(X_batch, dtype=np.float32)
		y_batch = np.array(y_batch, dtype=np.float32)

		self.index = end_index
		self.check_if_shuffle()

		return X_batch, y_batch, self.next_epoch


class DataSamplerMarkovChainRNN(DataSampler):
	def get_data(self, X, y, init_state, weights):
		'''
		This function returns the data that should be used for the current 
		batch.
		'''
		self.next_epoch = False

		if self.index == -1:
			self.index = 0
			self.num_instances = len(X)
			self.order = np.random.permutation(self.num_instances)

		# Get the indexes of the data to sample
		start_index = self.index
		end_index = start_index + self.batch_size

		if end_index > self.num_instances:
			end_index = self.num_instances

		batch_indexes = self.order[start_index:end_index]

		X_batch = list()
		y_batch = list()
		init_batch = list()
		weights_batch = list()

		for element in batch_indexes:
			X_batch.append(X[element])
			y_batch.append(y[element])
			init_batch.append(init_state[element])
			weights_batch.append(weights[element])

		X_batch = np.array(X_batch, dtype=np.float32)
		y_batch = np.array(y_batch, dtype=np.float32)
		init_batch = np.array(init_batch, dtype=np.float32)
		weights_batch = np.array(weights_batch, dtype=np.float32)

		self.index = end_index
		self.check_if_shuffle()

		return X_batch, y_batch, init_batch, weights_batch, self.next_epoch