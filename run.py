from helpers import *
#import sys
import numpy as np
from implementations import *
#import implementations
#from test_project1_public import *

DATASET_FOLDER_NAME = "dataset"
DATA_PATH = DATASET_FOLDER_NAME

def format_x(x):
	"""
	args: x : (N, D-1) matrix
	returns: tx : (N, D) matrix with offset term in first column
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
	
	print("Loading csv data...")
	x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)
	
	
	print("Loaded csv data.")
	y_train = (y_train == 1) # mapping -1 to 0 for mathematical consistency

	# I believe the first 20 columns are ids and shouldn't be present in the training data
	# I didn't check the other columns so there might be more
	x_train_format = x_train[:, 20 :] 
	# replacing nan with median values (we'll have to see if this makes sense)

	nan_indices = np.isnan(x_train_format)
	medians = np.nanmedian(x_train_format, axis=0)
	x_imputed = np.where(nan_indices, medians, x_train_format)
	tx_train = format_x(x_imputed)
	D = tx_train.shape[1]

	initial_w = np.zeros(D)
	max_iters = 300
	gamma = 0.0000000001

	print("Training...")

	# we'll choose the best method

	#w, loss = mean_squared_error_sgd(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=max_iters, gamma=gamma)
	w, loss = reg_logistic_regression(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=max_iters, gamma=gamma, lambda_=0.5)
	#w, loss = logistic_regression(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=max_iters, gamma=gamma)

	print(f"final weigths : {w}")
	print(f"final loss : {loss}")

	
	x_test_format = x_test[:, 20 :] 
	nan_indices = np.isnan(x_test_format)
	medians = np.nanmedian(x_test_format, axis=0)
	x_test_imputed = np.where(nan_indices, medians, x_test_format)
	tx_test = format_x(x_test_imputed)
	y_pred = predict(tx_test, w)
	
	#create_csv_submission(ids=train_ids, y_pred=y_pred, name='predictions')

	return

if __name__ =="__main__":
	main()