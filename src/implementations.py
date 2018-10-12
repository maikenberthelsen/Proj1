#all implementation goes here

import numpy as np
import datetime
from helpers import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):

	"""Gradient descent algorithm."""
	# Define parameters to store w and loss
	ws = [initial_w]
	losses = []
	w = initial_w

	for n_iter in range(max_iters):
		# compute loss, gradient
		grad, err = compute_gradient(y, tx, w)
		loss = 1/2*np.mean(err**2)

		# gradient w by descent update
		w = w - gamma * grad
		# store w and loss
		ws.append(w)
		losses.append(loss)

	#finds best parameters
	min_ind = np.argmin(losses)
	loss = losses[min_ind]
	w = ws[min_ind][:]


	return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

	"""Stochastic gradient descent algorithm."""
	ws = [initial_w]
	losses = []
	w = initial_w
	batch_size = 1

	#iterate max_iters times, where a small batch is picked on each iteration.
	#Don't understant whyyy we do this?
	for n_iter in range(max_iters):
		for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
			grad, err = compute_gradient(minibatch_y, minibatch_tx, w)
			loss = 1/2*np.mean(err**2)

			w = w - gamma*grad

			# store w and loss
			ws.append(w)
			losses.append(loss)
			print(loss)

	#finds best parameters
	min_ind = np.argmin(losses)
	loss = losses[min_ind]
	w = ws[min_ind][:]

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

	aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
	a = tx.T.dot(tx) + aI
	b = tx.T.dot(y)
	w = np.linalg.solve(a, b)

	e = y - tx.dot(w)
	loss = e.dot(e) / (2 * len(e))

	return w, loss


# tentative suggestions for logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):

	ws = [initial_w]
	losses = []
	w = initial_w

	for n_iter in range(max_iters):
		# compute prediction, loss, gradient
		#tx should maybe not be transposed
		# not transposed when using large X
		prediction = sigmoid(tx.dot(w))
		loss = -(y.T.dot(np.log(prediction)) + (1-y).T.dot(np.log(1-prediction))) + ((lambda_/2)*(np.linalg.norm(w)**2))
		gradient = tx.T.dot(prediction - y)

		# gradient w by descent update
		w = w - gamma * grad
		# store w and loss
		ws.append(w)
		losses.append(loss)

	#finds best parameters
	min_ind = np.argmin(losses)
	loss = losses[min_ind]
	w = ws[min_ind][:]
	return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	ws = [initial_w]
	losses = []
	w = initial_w

	for n_iter in range(max_iters):
		# compute prediction, loss, gradient
		#tx should maybe not be transposed
		# not transposed when using large X
		prediction = sigmoid(tx.dot(w))
		loss = -(y.T.dot(np.log(prediction)) + (1-y).T.dot(np.log(1-prediction))) + lambda_
		gradient = tx.T.dot(prediction - y) + (lambda_*np.linalg.norm(w))

		# gradient w by descent update
		w = w - gamma * grad
		# store w and loss
		ws.append(w)
		losses.append(loss)

	#finds best parameters
	min_ind = np.argmin(losses)
	loss = losses[min_ind]
	w = ws[min_ind][:]
	return w, loss


