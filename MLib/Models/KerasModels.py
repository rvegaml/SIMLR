import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from MLib.Core.Cells import SIR_Cell
from tensorflow.keras.layers import RNN


class MarkovChainRNN_Model(Model):
	def __init__(self, population):
		super(MarkovChainRNN_Model, self).__init__()
		
		cell = SIR_Cell(population)
		
		self.RNN = RNN(cell, return_state=True)
		
	def call(self, inputs, initial_state):
		tensor_initial = tf.convert_to_tensor(initial_state, dtype=tf.float32)
		output, state = self.RNN(inputs, initial_state=tensor_initial)
		
		return output, state
		
def main():
	return -1

if __name__ == '__main__':
	# Do nothing
	main()