'''
File: Infections.py
Author: Roberto Vega
Date: 14/08/2020
Email: rvega@ualberta.ca

Description:
	This file contains code for learning and doing inference with the following models:
		- SIR model (f(x), df(x), covariance)
'''
import numpy as np
from scipy.stats import norm
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import copy
from sklearn.preprocessing import OneHotEncoder

'''
------------------------------------------------------
Custom Machine Learning Models
------------------------------------------------------
'''
class RegLinearRegression():
    '''
    Our own implementation of Ridge regression that give us 
    more control over the regularization paramters.
    '''
    def __init__(self, alpha=1):
        '''
        alpha is the regularized applied to each weight
        '''
        self.alpha = alpha
        self.theta = None
        
    def fit(self, X, y):

        num_features = X.shape[1]

        lambda_matrix = np.diag(self.alpha)

        temp = np.dot(X.T, X) + lambda_matrix
        temp_inv = np.linalg.inv(temp)
        temp_2 = np.dot(X.T, y)

        theta = np.dot(temp_inv, temp_2)

        self.theta = theta
        
    def predict(self, X):
        return np.dot(X, self.theta)


'''
------------------------------------------------------
Code for dealing with Oxford Policies
------------------------------------------------------
'''
def get_daily_policies(country_data_features, features, encoder_dict):
    '''
    Function to compute the daily policy, which is a binary vector.
    '''
    daily_policy = list()
    num_days = len(country_data_features.iloc[:,0].values)

    for i in range(num_days):
        feature_vector = np.array([])

        for element in features[1:]:
            c_mode = country_data_features.loc[:, element].values[i]
            c_encoding = encoder_dict[element].transform([[c_mode]])

            feature_vector = np.concatenate([feature_vector, c_encoding[0]])
        daily_policy.append(feature_vector)

    daily_policy = np.array(daily_policy)
    
    return daily_policy

def create_policy_dict(policies_cardinality):
    encoder_dict = dict()

    for element in policies_cardinality.keys():
        print('Creating encoder for', element)
        c_encoder = OneHotEncoder(sparse=False)
        c_encoder.fit(np.reshape(np.arange(policies_cardinality[element]), (-1,1)))
        encoder_dict[element] = c_encoder
    
    return encoder_dict

'''
------------------------------------------------------
Loading and preprocessing code
------------------------------------------------------
'''

def extract_data(start_date, end_date, country_data, features):
    '''
    This function extracts and preprocess the new daily cases and deaths
    '''
    index_start_date = np.where(country_data['Date'].values == start_date)[0][0]
    index_end_date = np.where(country_data['Date'].values == end_date)[0][0]

    country_data_dates = country_data.iloc[index_start_date:index_end_date+1,]

    cases = country_data_dates.loc[:, 'ConfirmedCases'].values
    deaths = country_data_dates.loc[:, 'ConfirmedDeaths'].values

    # Compute the number of new cases and new deaths
    new_cases = cases[1:] - cases[0:-1]
    new_deaths = deaths[1:] - deaths[0:-1]

    # ---------------------------------
    # Preprocess the data to eliminate outliers, negative values, etc
    processed_cases = preprocess_timeseries(new_cases, threshold_outliers=500, min_index_missing=0)
    processed_deaths = preprocess_timeseries(new_deaths, threshold_outliers=100, min_index_missing=0)

    num_days = len(processed_cases)

    country_data_features = country_data_dates.loc[:, features[1:]]

    # Make sure that there are no nan values If there are, then replace them by
    # the last policy that was in place.
    for element in features[1:]:
        c_column = country_data_features.loc[:, element].values
        for i in range(num_days+1):
            if np.isnan(c_column[i]):
                if i == 0:
                    c_column[i] = 0
                else:
                    c_column[i] = c_column[i-1]

        country_data_features.assign(changing_features = np.array(c_column))
        
    dates = np.array(country_data_dates.loc[:, 'Date'].values[1:])
    
    return dates, processed_cases, processed_deaths, country_data_features.iloc[1:,:]

def create_SIR_data(population, processed_cases, processed_deaths, recovery_time=14):
    '''
    This function transforms the newly daily cases and deaths into S,I,R data.
    '''
    cum_region_cases = np.cumsum(processed_cases)
    cum_region_deaths = np.cumsum(processed_deaths)

    processed_recovered = np.zeros(len(processed_cases))

    deaths_eliminated = 0
    
    for i in range(len(cum_region_deaths) - recovery_time):
        possible_recovery = np.float(processed_cases[i])
        deaths_x_days = np.float(cum_region_deaths[i + recovery_time])

        if possible_recovery > 0:
            actual_recovered = possible_recovery - (deaths_x_days - deaths_eliminated)
            deaths_eliminated += (possible_recovery - actual_recovered)
            processed_recovered[i + recovery_time] = actual_recovered
        else:
            actual_recovered = 0

    all_region = np.vstack([processed_cases, processed_deaths, \
                            processed_recovered])

    # Create the SIR data
    S = [population - processed_cases[0] - processed_deaths[0] - processed_recovered[0]]
    I = [processed_cases[0]]
    R = [processed_deaths[0] + processed_recovered[0]]

    for i in range(len(processed_cases)-1):
        S.append(S[i] - processed_cases[i+1])
        I.append(I[i] + processed_cases[i+1] \
                 - processed_deaths[i+1] \
                 - processed_recovered[i+1])
        R.append(R[i] + processed_deaths[i+1] \
                 + processed_recovered[i+1])
    return S, I, R


def complete_missing_days(new_cases, min_index=20):
	'''
	Some countries do not report new unmber of cases in a given day(s) and they report
	the acumulated value on a posterior day. This preprocessing fills in the missing values
	by assuming that the cumulated number of cases is distributed evenly over the other days.
	'''
	preprocessed_cases = np.array(new_cases)
	
	for i in range(len(new_cases)):
		if (i > min_index) and (new_cases[i] == 0):
			counter = 1
			while True:
				if i+counter < len(new_cases):
					next_val = new_cases[i+counter]
					counter += 1
					if next_val != 0 :
						break
				else:
					next_val = 0
					break
			delta = next_val/counter                       
			for j in range(counter):
				preprocessed_cases[i+j] = delta
				
	return preprocessed_cases

def eliminate_negative_numbers(new_cases):
	'''
	We cannot have negative values in the new number of cases or deaths.
	When we face a negative number we will assume that the number of new cases over that day is the
	average of the last and next reported values
	'''
	preprocessed_deaths = np.array(new_cases)
	
	for i in range(len(new_cases)):
		if new_cases[i] < 0:
			counter = 1
			while True:
				past_val = new_cases[i-counter]
				if past_val != 0:
					break
				else:
					counter += 1

			counter = 1        
			while True:
				if i+counter < len(new_cases):
					future_val = new_cases[i+counter]
					if future_val != 0:
						break
					else:
						counter += 1
				else:
					future_val = past_val
					break
			preprocessed_deaths[i] = 0.5*(past_val + future_val)
	
	return preprocessed_deaths

def eliminate_outliers(new_cases, threshold=2000):
	'''
	There are days in which we see a sudden increase in the number of cases, for example
	it goes from 4,000 to 88,000 in one day. Such a change is very unlikely. We will consider
	this as an outlier if the change from one day to another is more than 3 times the current value.
	This change only applies when the number of cases is greater than threshold
	'''
	preprocessed_cases = np.array(new_cases)
	
	for i in range(len(new_cases)-1):
		if preprocessed_cases[i] > threshold:
			mu = np.mean(preprocessed_cases[i-10:i])
			sigma = np.std(preprocessed_cases[i-10:i])
			
			if preprocessed_cases[i] > mu+4*sigma:
				preprocessed_cases[i] = mu+4*sigma

	return preprocessed_cases

def complete_missing_recovered(timeseries_deaths, timeseries_recovered, threshold=50):
	new_daily_recovered = np.array(timeseries_recovered)
	
	for i in range(len(timeseries_recovered)):
		c_daily_recovered = timeseries_recovered[i]
		ratio_flag = False
		
		# If the coutry stopped reporting the recovered, then compute the ratio of deaths/recovered over the
		# last 4 weeks and use that to infer the number of recovered during the missing timesteps.
		if (c_daily_recovered == 0) and (i > threshold):
			if ratio_flag == False:
				ratio = np.mean(timeseries_deaths[i-28:i] / timeseries_recovered[i-28:i])
				ratio_flag = True
			if np.isnan(ratio):
				pass
			else:
				new_daily_recovered[i] = timeseries_deaths[i] / ratio
			
	return new_daily_recovered

def preprocess_timeseries(timeseries, threshold_outliers=2000, min_index_missing=20):
	'''
	This wrapper function preprocess a time series.
	
	threshold_outliers is the minimum number of new cases in a given day that can be considered
	as an outlier.
	
	min_index_missing is the the index where we will start looking for 0's in the time series. Usually
	there are many zeros at the beginning of the time series, so we might want to skip those.
	'''
	
	preprocessed_time_series = eliminate_negative_numbers(timeseries)
	preprocessed_time_series = complete_missing_days(preprocessed_time_series, min_index_missing)
	preprocessed_time_series = eliminate_outliers(preprocessed_time_series, threshold_outliers)
	
	return preprocessed_time_series

def date_to_str_Alberta(date):
    num_to_month = {'01':'Jan', '02':'Feb', '03':'Mar', '04':'Apr',\
                    '05':'May', '06':'Jun', '07':'Jul', '08':'Aug',\
                    '09':'Sep', '10':'Oct', '11':'Nov', '12':'Dec'}
    
    date_str = str(date)
    
    month = date_str[4:6]
    day = date_str[6:8]
    year = date_str[2:4]
    
    if day[0] == '0':
        day = day[1]
    
    return '-'.join([day, num_to_month[month], year])
    

'''
------------------------------------------------------
Fitting the SIR model
------------------------------------------------------
'''
def beta_gamma_solver(S, I, R):
	# Create the dataset for solving for Beta and Gamma
	S_next = np.array(S[1:])
	I_next = np.array(I[1:])
	R_next = np.array(R[1:])
	
	S_t = np.array(S[0:-1])
	I_t = np.array(I[0:-1])
	R_t = np.array(R[0:-1])
	
	P = S[0] + I[0] + R[0]
		
	delta_S = S_next - S_t
	delta_I = I_next - I_t
	delta_R = R_next - R_t
	
	# Create the matrices to solve the linear equations
	a=0; b=0; c=0; d=0; e=0
	
	num_elements = len(delta_S)
	
	for t in range(num_elements):
		a += 2 * (S_t[t]*I_t[t]/P)**2
		b -= S_t[t]*I_t[t]*I_t[t]/P
		c += 2 * I_t[t]*I_t[t]
		d -= S_t[t]*I_t[t]*(delta_S[t] - delta_I[t])/P
		e -= (delta_I[t] - delta_R[t]) * I_t[t]
		
	A_mat = np.array([
		[a, b],
		[b, c]
	])
	
	b_vec = np.array([[d],[e]])
	
	# Solve the system of linear equations
	solution = np.dot( np.linalg.inv(A_mat), b_vec )
	
	beta = solution[0,0]
	gamma = solution[1,0]
	
	return beta, gamma

def learn_beta_gamma_models(X_policy, beta_policy, gamma_policy, alpha):
    '''
    Create a ridge regression model to predict beta and gamma, given the policies.
    '''
    c_X_policy = np.array(X_policy)
    c_beta_policy = np.array(beta_policy)
    c_gamma_policy = np.array(gamma_policy)

    ridge_model_beta = RegLinearRegression(alpha)
    ridge_model_gamma = RegLinearRegression(alpha)

    ridge_model_beta.fit(c_X_policy, c_beta_policy)
    ridge_model_gamma.fit(c_X_policy, c_gamma_policy)
    
    return ridge_model_beta, ridge_model_gamma