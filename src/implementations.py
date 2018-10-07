#all implementation goes here

import numpy as np
import proj1_helpers


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def least_squares_GD(y, tx, initial_w, max_iters, gamma):

	"""Gradient descent algorithm."""
	# Define parameters to store w and loss
	ws = [initial_w]
	losses = []
	w = initial_w

	for n_iter in range(max_iters):
		# compute loss, gradient
		grad, err = compute_gradient(y, tx, w)
		1/2*np.mean(err**2)

		# gradient w by descent update
		w = w - gamma * grad
		# store w and loss
		ws.append(w)
		losses.append(loss)

	#finds best parameters
	min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
	loss = losses[min_row, min_col]
	w = [w0[min_row], w1[min_col]]

	return w, loss



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

	"""Stochastic gradient descent algorithm."""
	ws = [initial_w]
	losses = []
	w = initial_w

	#iterate max_iters times, where a small batch is picked on each iteration.
	#Don't understant whyyy we do this?
	for n_iter in range(max_iters):
		for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
			grad, loss = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
			w = w - gamma*grad

			# store w and loss
			ws.append(w)
			losses.append(loss)


	#finds best parameters
	min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
	loss = losses[min_row, min_col]
	w = [w0[min_row], w1[min_col]]


	return w, loss


def least_squares(y, tx):
	"""calculate the least squares solution."""

	a = tx.T.dot(tx)
	b = tx.T.dot(y)
	w = np.linalg.solve(a, b)

	e = y - tx.dot(w)
	loss = e.dot(e) / (2 * len(e))

	return w, loss


def ridge_regression(y, tx, lambda_):
    return
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return


def main():
	print("hello world")

	return 0;


main()
