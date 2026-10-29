from helpers import *
#import sys
import numpy as np
from implementations import *
#import implementations
#from test_project1_public import *

DATASET_FOLDER_NAME = "dataset"
DATA_PATH = DATASET_FOLDER_NAME
K_FOLD = 5
CV_SEED = 12
MAX_ITERS = 300
GAMMA = 0.02
LAMBDA = 0.001

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
	return np.where(tx.dot(w) >= 0, 1, -1)

def impute_and_standardize(x_train, x_other):
	"""Impute with training medians and standardize using training statistics."""
	medians = np.nanmedian(x_train, axis=0)
	x_train = np.where(np.isnan(x_train), medians, x_train)
	x_other = np.where(np.isnan(x_other), medians, x_other)

	means = np.mean(x_train, axis=0)
	scales = np.std(x_train, axis=0)
	scales[scales == 0] = 1
	return (x_train - means) / scales, (x_other - means) / scales


def build_k_indices(y, k_fold, seed):
	"""Build seeded, shuffled folds while keeping every sample in one fold."""
	if k_fold < 2 or k_fold > len(y):
		raise ValueError("k_fold must be between 2 and the number of samples")

	indices = np.random.RandomState(seed).permutation(len(y))
	return np.array_split(indices, k_fold)


def cross_validation(y, x, k_indices, k, lambda_, max_iters, gamma):
	"""Fit regularized logistic regression and return accuracy on fold k."""
	if k < 0 or k >= len(k_indices):
		raise ValueError("k must identify a fold in k_indices")

	validation_indices = k_indices[k]
	training_indices = np.concatenate(
		[fold for fold_index, fold in enumerate(k_indices) if fold_index != k]
	)
	x_train = x[training_indices]
	x_validation = x[validation_indices]
	y_train = y[training_indices]
	y_validation = y[validation_indices]

	x_train, x_validation = impute_and_standardize(x_train, x_validation)
	tx_train = format_x(x_train)
	tx_validation = format_x(x_validation)

	initial_w = np.zeros(tx_train.shape[1])
	w, _ = reg_logistic_regression(
		y=y_train,
		tx=tx_train,
		lambda_=lambda_,
		initial_w=initial_w,
		max_iters=max_iters,
		gamma=gamma,
	)
	predictions = (tx_validation.dot(w) >= 0).astype(float)
	return np.mean(predictions == y_validation)


def main():
	
	print("Loading csv data...")
	x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)
	
	
	print("Loaded csv data.")
	y_train = (y_train == 1).astype(float)

	print(f"Running {K_FOLD}-fold cross-validation...")
	k_indices = build_k_indices(y_train, K_FOLD, CV_SEED)
	fold_accuracies = [
		cross_validation(
			y_train, x_train, k_indices, k, LAMBDA, MAX_ITERS, GAMMA
		)
		for k in range(K_FOLD)
	]
	for fold, accuracy in enumerate(fold_accuracies, start=1):
		print(f"Fold {fold} validation accuracy: {accuracy:.4f}")
	print(f"Mean validation accuracy: {np.mean(fold_accuracies):.4f}")

	x_train, x_test = impute_and_standardize(x_train, x_test)
	tx_train = format_x(x_train)
	tx_test = format_x(x_test)
	initial_w = np.zeros(tx_train.shape[1])
	final_w, loss = reg_logistic_regression(
		y=y_train,
		tx=tx_train,
		lambda_=LAMBDA,
		initial_w=initial_w,
		max_iters=MAX_ITERS,
		gamma=GAMMA,
	)
	print(f"Final training loss: {loss}")

	y_pred = predict(tx_test, final_w)
	
	create_csv_submission(ids=test_ids, y_pred=y_pred, name='predictions.csv')

	return

if __name__ =="__main__":
	main()