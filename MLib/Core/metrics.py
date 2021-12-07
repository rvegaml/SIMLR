import numpy as np


def MAPE(y_real, y_predicted, aggregate):
	'''
	MAPE: mean absolute percentage error
	'''
	error = y_real - y_predicted
	absolute_error = np.abs(error)

	percentage_error = 100*absolute_error/y_real
	
	MAPE = aggregate(percentage_error)
	average_deviation = np.std(percentage_error)

	return MAPE, average_deviation

def MASE(y_real, y_predicted, r=14):
	'''
	MASE: Mean absoulte scaled error. r is the seasonality
	'''
	error = y_real - y_predicted
	absolute_error = np.abs(error)
	sum_absolute_error = np.sum(absolute_error)

	real_minus_r = np.zeros(len(y_real))
	real_minus_r[r:] = y_real[0:-r]

	sum_scaled = np.sum(np.abs(y_real - real_minus_r))

	return sum_absolute_error/sum_scaled

def MSE(y_real, y_predicted):
	'''
	MSE: Mean squared error
	'''
	error = y_real - y_predicted
	sse = np.dot(error, error)

	T = len(y_real)

	return sse/T