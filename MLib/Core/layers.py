import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import glorot_normal

class Identity(layers.Layer):
	'''
	Identity layer. It returns a copy of the input.
	'''

	# According to the TensorFlow documentation, it's a good practice to add this function
	def __init__(self, **kwargs):
		super(Identity, self).__init__(**kwargs)

	# According to the TensorFlow documentation, it's a good practice to add this function
	def get_config(self):
		config = super(Identity, self).get_config()
		return config

	def call(self, inputs):
		return tf.identity(inputs)

class LinearRegLayer(layers.Layer):
	'''
	A simple linear layer that penalizes deviations from the initial weights
	'''

	# According to the TensorFlow documentation, it's a good practice to add this function
	def __init__(self, units=10, rate=.01, init_weights=[], **kwargs):
		super(LinearRegLayer, self).__init__(**kwargs)
		self.units = units
		self.rate = rate
		self.init_weights = init_weights

	# According to the TensorFlow documentation, it's a good practice to add this function
	def get_config(self):
		config = super(LinearRegLayer, self).get_config()
		return config


	def build(self, input_shape):
		# Get the number of dimensions of the data
		num_dim = input_shape[-1]

		# Build the actual weights
		self.w = self.add_weight(shape=(num_dim, self.units),
			initializer='random_normal',
			trainable=True)
		self.b = self.add_weight(shape=(self.units,),
			initializer='random_normal',
			trainable=True)

	def call(self, inputs):
		current_weights = self.weights

		current_w = current_weights[0]
		current_b = current_weights[1]

		init_w = np.array(self.init_weights[0])
		init_b = np.array(self.init_weights[1])

		diff_w = K.sum(K.square(current_w - init_w))
		diff_b = K.sum(K.square(current_b - init_b))

		total_diff = diff_w + diff_b

		self.add_loss(self.rate * total_diff)

		return tf.matmul(inputs, self.w) + self.b

class LinearLayer(layers.Layer):
	'''
	A simple linear layer that penalizes deviations from the initial weights
	'''

	# According to the TensorFlow documentation, it's a good practice to add this function
	def __init__(self, units=10, **kwargs):
		super(LinearLayer, self).__init__(**kwargs)
		self.units = units

	# According to the TensorFlow documentation, it's a good practice to add this function
	def get_config(self):
		config = super(LinearLayer, self).get_config()
		return config


	def build(self, input_shape):
		# Get the number of dimensions of the data
		num_dim = input_shape[-1]

		# Build the actual weights
		self.w = self.add_weight(shape=(num_dim, self.units),
			initializer='random_normal',
			trainable=True)
		self.b = self.add_weight(shape=(self.units,),
			initializer='random_normal',
			trainable=True)

	def call(self, inputs):

		return tf.matmul(inputs, self.w) + self.b

class ConvLayer(layers.Layer):
	'''
	Layer that computes the 2D convolution and penalizes deviations from weights.
	'''

	def __init__(self, size=[3,3], num_filters=32, gate=tf.nn.relu, 
		stride=[1,1,1,1], padding='SAME', **kwargs):
	
		super(ConvLayer, self).__init__(**kwargs)

		self.size = size
		self.num_filters = num_filters
		self.gate = gate
		self.stride = stride
		self.padding = padding

	# According to the TensorFlow documentation, it's a good practice to add this function
	def get_config(self):
		config = super(ConvLayer, self).get_config()
		return config

	def build(self, input_shape):
		# Get the number of dimensions of the data
		dim_in = input_shape[-1]
		filter_height = self.size[0]
		filter_width = self.size[1]

		# Build the actual weights
		self.w = self.add_weight(shape=(filter_height, filter_width, dim_in, self.num_filters),
			initializer=glorot_normal(),
			trainable=True)
		self.b = self.add_weight(shape=(self.num_filters,),
			initializer=glorot_normal(),
			trainable=True)

	def call(self, inputs):
		
		x = tf.nn.conv2d(inputs, filters=self.w, strides=self.stride, padding=self.padding)
		x = tf.add(x, self.b)

		return self.gate(x)

class RegConvLayer(layers.Layer):
	'''
	Layer that computes the 2D convolution and penalizes deviations from weights.
	'''

	def __init__(self, size=[3,3], num_filters=32, gate=tf.nn.relu, rate=.01, 
		init_weights=[], stride=[1,1,1,1], padding='SAME', **kwargs):

		super(RegConvLayer, self).__init__(**kwargs)
		self.rate = rate
		self.init_weights = init_weights
		self.size = size
		self.num_filters = num_filters
		self.gate = gate
		self.stride = stride
		self.padding = padding

	# According to the TensorFlow documentation, it's a good practice to add this function
	def get_config(self):
		config = super(RegConvLayer, self).get_config()
		return config

	def build(self, input_shape):
		# Get the number of dimensions of the data
		dim_in = input_shape[-1]
		filter_height = self.size[0]
		filter_width = self.size[1]

		# Build the actual weights
		self.w = self.add_weight(shape=(filter_height, filter_width, dim_in, self.num_filters),
			initializer=glorot_normal(),
			trainable=True)
		self.b = self.add_weight(shape=(self.num_filters,),
			initializer=glorot_normal(),
			trainable=True)

	def call(self, inputs):
		current_weights = self.weights

		current_w = current_weights[0]
		current_b = current_weights[1]

		init_w = np.array(self.init_weights[0])
		init_b = np.array(self.init_weights[1])

		diff_w = K.sum(K.square(current_w - init_w))
		diff_b = K.sum(K.square(current_b - init_b))

		total_diff = diff_w + diff_b

		self.add_loss(self.rate * total_diff)

		x = tf.nn.conv2d(inputs, self.w, strides=self.stride, padding=self.padding)
		x = tf.add(x, self.b)

		return self.gate(x)

class RegTransposeConvLayer(layers.Layer):
	'''
	Layer that computes the 2D convolution and penalizes deviations from weights.
	'''

	def __init__(self, size=[3,3], num_filters=32, gate=tf.nn.relu, rate=.01, 
		init_weights=[], stride=[1,1,1,1], padding='SAME', **kwargs):

		super(RegTransposeConvLayer, self).__init__(**kwargs)
		self.rate = rate
		self.init_weights = init_weights
		self.size = size
		self.num_filters = num_filters
		self.gate = gate
		self.stride = stride
		self.padding = padding
		self.output_shape = None

	# According to the TensorFlow documentation, it's a good practice to add this function
	def get_config(self):
		config = super(RegTransposeConvLayer, self).get_config()
		return config

	def build(self, input_shape):
		# Get the number of dimensions of the data
		batch = input_shape[0]
		dim_in = input_shape[-1]
		filter_height = self.size[0]
		filter_width = self.size[1]

		new_height = deconv_output_length(height, filter_height, padding, strides[1])
		new_width = deconv_output_length(width, filter_width, padding, strides[2])

		self.output_shape = tf.convert_to_tensor([batch, new_height, new_width, self.num_filters])

		# Build the actual weights
		self.w = self.add_weight(shape=(filter_height, filter_width, self.num_filters, dim_in),
			initializer=glorot_normal(),
			trainable=True)
		self.b = self.add_weight(shape=(self.num_filters,),
			initializer=glorot_normal(),
			trainable=True)

	def call(self, inputs):
		current_weights = self.weights

		current_w = current_weights[0]
		current_b = current_weights[1]

		init_w = np.array(self.init_weights[0])
		init_b = np.array(self.init_weights[1])

		diff_w = K.sum(K.square(current_w - init_w))
		diff_b = K.sum(K.square(current_b - init_b))

		total_diff = diff_w + diff_b

		self.add_loss(self.rate * total_diff)

		x = tf.nn.conv2d_transpose(inputs, self.w,self.output_shape,
			strides=self.stride, padding=self.padding)
		x = tf.add(x, self.b)

		return self.gate(x)


class MyReshape(layers.Layer):
	def __init__(self, target_shape, **kwargs):
		super(MyReshape, self).__init__(**kwargs)
		self.target_shape = target_shape

	# According to the TensorFlow documentation, it's a good practice to add this function
	def get_config(self):
		config = super(MyReshape, self).get_config()
		return config

	def call(self, inputs):
		reshaped = tf.reshape(inputs, self.target_shape)

		return reshaped

# # -----------------------------------------------------------------------------------------
# # The following functions used to work on TensorFlow 1.XX
# # Create the custom 3D-Layer
# def Convolution_3D(name, label, inputs, kernel_size, channels_in, channels_out, transfer, 
# 	strides=[1,1,1], padding='SAME', initializer_W=None, initializer_b=None, reuse=False):
# 	with tf.variable_scope(name, reuse=reuse):
# 		with tf.variable_scope(label, reuse=reuse):
# 			W = tf.get_variable('W', [kernel_size, kernel_size, kernel_size, channels_in, channels_out], 
# 				initializer=initializer_W)
# 			b = tf.get_variable('bias', [channels_out], initializer=initializer_b)

# 	# The first and last elemnts of strides should alwasys be 1
# 	c_strides = [1] + list(strides) + [1]

# 	# Perform the 3D convolution
# 	z_hat = tf.nn.conv3d(inputs, W, strides=c_strides, padding=padding)
# 	# Add the bias
# 	z_hat = tf.nn.bias_add(z_hat, b)
# 	# Apply the transfer function
# 	y_hat = transfer(z_hat)

# 	return W, b, z_hat, y_hat

# def Up_Convolution_3D(name, label, inputs, kernel_size, channels_in, channels_out, 
# 	strides=[2,2,2], padding='SAME', initializer=None, reuse=False):

# 	with tf.variable_scope(name, reuse=reuse):
# 		with tf.variable_scope(label, reuse=reuse):
# 			W = tf.get_variable('W', [kernel_size, kernel_size, kernel_size, channels_out, channels_in], 
# 				initializer=initializer)

# 	# The first and last elemnts of strides should alwasys be 1
# 	c_strides = [1] + list(strides) + [1]

# 	# Extract the shape of the inputs
# 	inputs_size = tf.shape(inputs)

# 	batch = inputs_size[0] 
# 	depth = inputs_size[1]
# 	height = inputs_size[2]
# 	width = inputs_size[3] 
# 	in_channels = inputs_size[4]


# 	# Compute the shape after the de-convolution
# 	new_depth = deconv_output_length(depth, kernel_size, padding, strides[0])
# 	new_height = deconv_output_length(height, kernel_size, padding, strides[1])
# 	new_width = deconv_output_length(width, kernel_size, padding, strides[2])

# 	output_shape = tf.convert_to_tensor([batch, new_depth, new_height, new_width, channels_out])

# 	# Apply the deconvolution
# 	z_hat = tf.nn.conv3d_transpose(inputs, W, output_shape, strides=c_strides, padding=padding)

# 	return W, z_hat

# def Fully_Connected(name, label, inputs, dim_in, dim_out, transfer, reuse=False):
# 	with tf.variable_scope(name, reuse=reuse):
# 		with tf.variable_scope(label, reuse=reuse):
# 			W = tf.get_variable('W', [dim_in, dim_out])
# 			b = tf.get_variable('b', [dim_out])

# 	z_hat = tf.matmul(inputs, W) + b
	
# 	y_hat = transfer(z_hat)

# 	return W, b, z_hat, y_hat

# def Convolution_2D(name, label, inputs, kernel_size, channels_in, channels_out, transfer, 
# 	strides=[1,1],  padding='SAME', initializer_W=None, initializer_b=None, reuse=False):
# 	'''
# 	This layer computes the 2D standard convolution.

# 	Arguments:
# 	name: Name of the network.
# 	label: Name of this particular layer
# 	inputs: A tf.placeholder containing the inputs: [num_images, height, width, channels]
# 	kernel_size: An int specifing the size of the kxk kernel
# 	channels_in: Number of channels of the input
# 	channels_out: Number of filters to create
# 	transfer: Transfer function to use.
# 	strides: The first and the last elements are always 1. The elements in the middle are the
# 		y and x steps.
# 	'''
# 	with tf.variable_scope(name, reuse=reuse):
# 		with tf.variable_scope(label, reuse=reuse):
# 			W = tf.get_variable('W', [kernel_size, kernel_size, channels_in, channels_out], 
# 				initializer=initializer_W)
# 			b = tf.get_variable('bias', [channels_out], initializer=initializer_b)

# 	# The first and last elemnts of strides should alwasys be 1
# 	c_strides = [1] + list(strides) + [1]

# 	# Compute the convolution
# 	z_hat = tf.nn.conv2d(inputs, W, c_strides, padding)
# 	# Add the bias
# 	z_hat = tf.nn.bias_add(z_hat, b)
# 	# Apply the transfer function
# 	y_hat = transfer(z_hat)

# 	return W, b, z_hat, y_hat

def deconv_output_length(input_length, filter_size, padding, stride):
	"""This function was adapted from Keras
		  Determines output length of a transposed convolution given input length.
	Arguments:
	  input_length: integer.
	  filter_size: integer.
	  padding: one of "same", "valid", "full".
	  stride: integer.
	Returns:
	  The output length (integer).
	"""
	if input_length is None:
		return None

	output_length = input_length * stride

	if padding == 'VALID':
		output_length = output_length + max(filter_size - stride, 0)

	return output_length

def main():
	return -1

if __name__ == '__main__':
	# Do nothing
	main()