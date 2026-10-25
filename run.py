from helpers import *
import sys
import numpy as np
from implementations import *
import implementations
from test_project1_public import *
DATASET_FOLDER_NAME = "dataset"
DATA_PATH = DATASET_FOLDER_NAME
def format_x(x):
	"""
	args: x : (N, D-1) matrix
	returns: tx : (N, D) matrix with offset in first column
	"""
	offsets = np.ones(x.shape[0])
	return np.column_stack([offsets, x])

def predict(tx, w):
	"""
	args: tx : (N,D) testing set
		  w : (D,) weights acquired after training

	returns : y (N,) the vector of predictions
	"""
	return np.sign(tx.dot(w))

def main():
	"""
	y = np.array([0.1, 0.3, 0.5])
	tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
	initial_w = np.array([0.5, 1.0])
	test_mean_squared_error_sgd(implementations, y=y, tx=tx, initial_w=initial_w)
	test_least_squares(implementations, y, tx)
	test_ridge_regression_lambda1(implementations, y, tx)
	test_logistic_regression(implementations, y, tx, initial_w)
	test_reg_logistic_regression(implementations, y, tx, initial_w)
	"""
	
	print("Loading csv data...")
	x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)
	#print(x_train[0])
	#print(train_ids[0])
	
	print("Loaded csv data.")
	y_train = (y_train == 1)
	# cleaning nans in x
	x_train_format = x_train[:, 20 :] # the first 20 columns are id
	nan_indices = np.isnan(x_train_format)
	medians = np.nanmedian(x_train_format, axis=0)
	x_imputed = np.where(nan_indices, medians, x_train_format)
	tx_train = format_x(x_imputed)
	#print(f"tx train[0] : {tx_train[0]}")
	D = tx_train.shape[1]
	#print(np.max(np.abs(tx_train)))
	initial_w = np.zeros(D)
	max_iters = 300
	gamma = 0.0000000001
	print("Training...")
	#w, loss = mean_squared_error_sgd(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=max_iters, gamma=gamma)
	w, loss = reg_logistic_regression(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=max_iters, gamma=gamma, lambda_=0.5)
	#w, loss = logistic_regression(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=10000, gamma=0.005)
	print(f"final weigths : {w}")
	print(f"final loss : {loss}")

	x_test_format = x_test[:, 20 :] # the first 20 columns are id
	nan_indices = np.isnan(x_test_format)
	medians = np.nanmedian(x_test_format, axis=0)
	x_test_imputed = np.where(nan_indices, medians, x_test_format)
	tx_test = format_x(x_test_imputed)
	y_test = predict(tx_test, w)
	print(y_test[20:])
	
	
	return

if __name__ =="__main__":
	main()