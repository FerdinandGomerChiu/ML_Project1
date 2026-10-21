import numpy as np

def compute_loss(y, tx, w):
	"""Calculate the loss using MSE.

	Args:
		y: numpy array of shape=(N, )
		tx: numpy array of shape=(N,2)
		w: numpy array of shape=(2,). The vector of model parameters.

	Returns:
		the value of the loss (a scalar), corresponding to the input parameters w.
	"""
	# ***************************************************
	# INSERT YOUR CODE HERE
	# TODO: compute loss by MSE
	# ***************************************************
	N = len(y)
	error = y - tx @ w
	return 1/(2*N) * (np.transpose(error) @ error)

def compute_gradient(y, tx, w):
	"""Computes the gradient at w.

	Args:
		y: numpy array of shape=(N, )
		tx: numpy array of shape=(N,2)
		w: numpy array of shape=(2, ). The vector of model parameters.

	Returns:
		An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
	"""
	# ***************************************************
	# INSERT YOUR CODE HERE
	# TODO: compute gradient vector
	# ***************************************************
	#raise NotImplementedError
	N = len(y)
	# print(tx.shape)
	error = y - (tx @ w)
	
	return (-1/N) * (np.transpose(tx) @ error)

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
	"""The Gradient Descent (GD) algorithm.

	Args:
		y: numpy array of shape=(N, )
		tx: numpy array of shape=(N,2)
		initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
		max_iters: a scalar denoting the total number of iterations of GD
		gamma: a scalar denoting the stepsize

	Returns:
		losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
		ws: a list of length max_iters + 1 containing the model parameters as numpy arrays of shape (2, ),
			for each iteration of GD (as well as the final weights)
	"""
	# Define parameters to store w and loss
	ws = [initial_w]
	losses = []
	w = initial_w
	loss=0
	for n_iter in range(max_iters):
		
		# compute gradient and loss
		loss = compute_loss(y,tx,w)

		g = compute_gradient(y, tx, w)
	   
		# update w by gradient
		w = w - gamma * g

		# store w and loss
		ws.append(w)
		losses.append(loss)
		print(
			"GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
				bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
			)
		)

	return w, loss # we only return the last weight and corresponding loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
	"""
	Generate a minibatch iterator for a dataset.
	Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
	Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
	Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

	Example:

	 Number of batches = 9

	 Batch size = 7                              Remainder = 3
	 v     v                                         v v
	|-------|-------|-------|-------|-------|-------|---|
		0       7       14      21      28      35   max batches = 6

	If shuffle is False, the returned batches are the ones started from the indexes:
	0, 7, 14, 21, 28, 35, 0, 7, 14

	If shuffle is True, the returned batches start in:
	7, 28, 14, 35, 14, 0, 21, 28, 7

	To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
	8, 28, 16, 38, 14, 0, 22, 28, 9

	This way batches might overlap, but the returned batches are slightly more representative.

	Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

	Example of use :
	for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
		<DO-SOMETHING>
	"""
	data_size = len(y)  # NUmber of data points.
	batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
	max_batches = int(
		data_size / batch_size
	)  # The maximum amount of non-overlapping batches that can be extracted from the data.
	remainder = (
		data_size - max_batches * batch_size
	)  # Points that would be excluded if no overlap is allowed.

	if shuffle:
		# Generate an array of indexes indicating the start of each batch
		idxs = np.random.randint(max_batches, size=num_batches) * batch_size
		if remainder != 0:
			# Add an random offset to the start of each batch to eventually consider the remainder points
			idxs += np.random.randint(remainder + 1, size=num_batches)
	else:
		# If no shuffle is done, the array of indexes is circular.
		idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

	for start in idxs:
		start_index = start  # The first data point of the batch
		end_index = (
			start_index + batch_size
		)  # The first data point of the following batch
		yield y[start_index:end_index], tx[start_index:end_index]
		

def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma):
	"""The Stochastic Gradient Descent algorithm (SGD).

	Args:
		y: numpy array of shape=(N, )
		tx: numpy array of shape=(N,2)
		initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
		batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
		max_iters: a scalar denoting the total number of iterations of SGD
		gamma: a scalar denoting the stepsize

	Returns:
		losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
		ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
	"""

	# Define parameters to store w and loss
	ws = [initial_w]
	losses = []
	w = initial_w
	num_batches=1
	loss= 0
	batch_size = 1
	for n_iter in range(max_iters):
		# ***************************************************
		# INSERT YOUR CODE HERE
		# TODO: implement stochastic gradient descent.
		# ***************************************************

		# raise NotImplementedError
		for minibatch_y, minibatch_tx in batch_iter(y=y, tx=tx, batch_size=batch_size, num_batches=num_batches):

			loss = compute_loss(y=minibatch_y, tx=minibatch_tx, w=w)
			g = compute_gradient(y=minibatch_y, tx=minibatch_tx, w=w)
			w = w - gamma * g

		# raise NotImplementedError

		# store w and loss
		ws.append(w)
		losses.append(loss)
		
		print(
			"SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
				bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
			)
		)
	return w, loss # we only return the last weight and corresponding loss

def least_squares(y, tx):
	"""Calculate the least squares solution.
	   returns mse, and optimal weights.

	Args:
		y: numpy array of shape (N,), N is the number of samples.
		tx: numpy array of shape (N,D), D is the number of features.

	Returns:
		w: optimal weights, numpy array of shape(D,), D is the number of features.
		mse: scalar.

	>>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
	(array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
	"""
	a = tx.T.dot(tx)
	b = tx.T.dot(y)
	w = np.linalg.solve(a, b)
	mse = compute_loss(y, tx, w)
	return w, mse

def ridge_regression(y, tx, lambda_):
	"""implement ridge regression.

	Args:
		y: numpy array of shape (N,), N is the number of samples.
		tx: numpy array of shape (N,D), D is the number of features.
		lambda_: scalar.

	Returns:
		w: optimal weights, numpy array of shape(D,), D is the number of features.

	>>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
	array([ 0.21212121, -0.12121212])
	>>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
	array([0.03947092, 0.00319628])
	"""
	aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
	a = tx.T.dot(tx) + aI
	b = tx.T.dot(y)
	return np.linalg.solve(a, b)