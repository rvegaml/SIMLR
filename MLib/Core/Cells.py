import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class SIR_Cell(layers.Layer):
	def __init__(self, population, **kwargs):
		'''
		Initialization of the Cell. state_size and output_size are mandatory elements.
		Since this is a simple Markov Chain, there is no hidden state
		The size of the output is 3 (S, I, R)
		'''
		self.state_size = 3
		self.output_size = 3
		self.Population = population
		# self.gamma = tf.constant(gamma, shape=(1,), dtype=tf.float32, name='gamma')

		super(SIR_Cell, self).__init__(**kwargs)

	def build(self, input_shape):
		self.beta = self.add_weight(shape=(1,),
			initializer='uniform', name='beta')
		self.gamma = self.add_weight(shape=(1,),
			initializer='uniform', name='gamma')
		self.build = True

	def call(self, inputs, states):
		'''
		inputs are (batch, input_size)
		states are (batch, state_size)
		'''
		S = states[0][:,0]
		I = states[0][:,1]
		R = states[0][:,2]
		
		S_next = S - inputs[:,0]*inputs[:,1]*S*I*self.beta/self.Population
		I_next = I + inputs[:,0]*inputs[:,1]*S*I*self.beta/self.Population - inputs[:,1]*self.gamma*I
		R_next = R + inputs[:,1]*self.gamma*I

		prediction = tf.stack([S_next, I_next, R_next], axis=1)
		next_state = tf.stack([S_next, I_next, R_next], axis=1)
		
		return prediction, next_state

def main():
	return -1

if __name__ == '__main__':
	# Do nothing
	main()