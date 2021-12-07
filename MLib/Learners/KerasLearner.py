'''
Code that allows an easy way for training Keras models
'''
import numpy as np
import pickle
import tensorflow as tf
from MLib.Models.KerasModels import MarkovChainRNN_Model
from MLib.Core.data_loaders import DataSamplerMarkovChainRNN
from tensorflow.keras.losses import MSE, MeanSquaredError

# tf.keras.backend.set_floatx('float32')

class Learner():
	'''
	Base learner that contains common functions for most of the learners
	'''
	def __init__(self):
		self.model = None

	def get_training_parameters(self, params):
		'''
		Function that returns the parameersneeded for the training procedure
		'''

		if 'epochs' in params:
			epochs = params['epochs']
		else:
			epochs = 10000

		if 'learning_rate' in params:
			lr = params['learning_rate']
		else:
			lr = 1E-5

		if 'optimizer' in params:
			optimizer = params['optimizer']
		else:
			optimizer = tf.keras.optimizers.Adam(lr)

		if 'loss_metric' in params:
			loss_metric = params['loss_metric']
		else:
			loss_metric = tf.keras.metrics.Mean()

		if 'batch_size' in params:
			batch_size = params['batch_size']
		else:
			batch_size = 50

		return epochs, lr, optimizer, loss_metric, batch_size

	def predict(self, X):
		return(self.model(X))

	def score(self, X, y, thresh=0.5):
		predictions = self.predict(X).numpy()

		predictions = np.reshape(predictions, (-1))
		predictions_bool = predictions > thresh

		ground_truth = np.reshape(y, (-1))
		ground_truth_bool = ground_truth > thresh

		accuracy = np.mean(np.equal(predictions_bool, ground_truth_bool))

		return accuracy

	def get_weights(self):
		return self.model.get_weights()


class MarkovChainRNN(Learner):
	def __init__(self, population):
		'''
			Initialize the Kalman Markov Chain Model.
		'''
		self.model = MarkovChainRNN_Model(population)
	
	
	def train(self, X, y, initial_state, params={}, weights=None):
		# Get the training parameters
		epochs, lr, optimizer, loss_metric, batch_size = self.get_training_parameters(params)
		loss_t_1 = np.inf
	
		# Initialize a data sampler to feed the data
		data_sampler = DataSamplerMarkovChainRNN(batch_size)
		next_epoch = False

		loss_function = MeanSquaredError()
		# Perform the actual training
		for epoch in range(epochs):
			# print('Start of epoch %d' % (epoch,))
			while next_epoch == False:
				with tf.GradientTape() as tape:
					X_batch, y_batch, init_batch, weights_batch, next_epoch = \
						data_sampler.get_data(X, y, initial_state, weights)
					
					y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
					y_hat, h_hat = self.model(X_batch, init_batch)
					
					loss = loss_function(y_true=y_batch, y_pred=y_hat, sample_weight=weights_batch)
					
				grads = tape.gradient(loss, self.model.trainable_weights)

				optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

				loss_metric(loss)

			next_epoch = False
			if epoch % 2 == 0:
				loss_t = loss_metric.result().numpy()
				print('Epoch ', epoch, ' Loss: ', loss_t, end='\r')

				if np.abs(loss_t - loss_t_1) < 1E-4:
					print('\n')
					print('Ending condition met. Weights converged')
					break

				loss_t_1 = loss_t

			next_epoch = False

	def predict(self, X, initial_state):
		return(self.model(X, initial_state))

	def score(self, X, y):
		print('Function not implemented for this model')

def main():
	return -1

if __name__ == '__main__':
	# Do nothing
	main()