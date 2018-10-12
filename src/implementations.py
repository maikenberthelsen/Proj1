#all implementation goes here

import numpy as np
import datetime
from proj1_helpers import *
import matplotlib.pyplot as plt


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

def sigmoid(t):
	"""apply sigmoid function on t."""
	return np.exp(t)/(1+ np.exp(t))

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


def run_gradient_descent(y, x):

	y, tx = build_model_data(y,x)

	max_iters = 150
	gamma = 0.1 #Ikke høyere enn 0.15, da konvergerer det ikke
	#initial_w = np.zeros(tx.shape[1])
	initial_w = [-0.3428, 0.01885391, -0.26018961, -0.22812764, -0.04019317, -0.00502791,
		0.32302178, -0.01464156, 0.23543933, 0.00973278, -0.0048371, -0.13453445,
  		0.13354281, -0.0073677, 0.22358728, 0.01132979, -0.00372824, 0.25739398,
  		0.02175267,  0.01270975,  0.12343641, -0.00613063, -0.09086221, -0.20328519,
  		0.05932847, 0.049829, 0.05156299, -0.01579745, -0.00793358, -0.00886158, -0.10660545]

	start_time = datetime.datetime.now()
	gd_w, gd_loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
	end_time = datetime.datetime.now()
	exection_time = (end_time - start_time).total_seconds()

	print('GD\n time = ', exection_time)
	print('w = ', gd_w)
	print('MSE = ', gd_loss)

	return gd_w, gd_loss

def run_stochastic_gradient_descent(y,x):

	y, tx = build_model_data(y,x)

	max_iters = 100
	gamma = 0.01 #Ikke høyere enn 0.15, da konvergerer det ikke
	batch_size = 1

	initial_w = [-0.3428, 0.01885391, -0.26018961, -0.22812764, -0.04019317, -0.00502791,
		0.32302178, -0.01464156, 0.23543933, 0.00973278, -0.0048371, -0.13453445,
  		0.13354281, -0.0073677, 0.22358728, 0.01132979, -0.00372824, 0.25739398,
  		0.02175267,  0.01270975,  0.12343641, -0.00613063, -0.09086221, -0.20328519,
  		0.05932847, 0.049829, 0.05156299, -0.01579745, -0.00793358, -0.00886158, -0.10660545]

	start_time = datetime.datetime.now()
	sgd_w, sgd_loss = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
	end_time = datetime.datetime.now()
	exection_time = (end_time - start_time).total_seconds()

	print('SGD\n time = ', exection_time)
	print('w = ', sgd_w)
	print('MSE = ', sgd_loss)

	return sgd_w, sgd_loss


def run_least_square(y,x):
	degree = 5
	tx = build_poly(x,degree)

	ls_w, ls_loss = least_squares(y, tx)

	return ls_w, ls_loss, degree


def run_ridge_regression(y,x):
	lambda_ = 0.003
	degree = 4

	tx = build_poly(x,degree)

	rr_w, rr_loss = ridge_regression(y, tx, lambda_)

	return rr_w, rr_loss, degree

def tune_ridge_regression(y,x):
	
	lambdas = np.logspace(-5,0,20)
	degree = 4
	ratio = 0.8

	x_tr, x_te, y_tr, y_te = split_data(y, x, ratio)

	tx_tr = build_poly(x_tr,degree)
	tx_te = build_poly(x_te,degree)

	rmse_tr = []
	rmse_te = []

	for ind, lambda_ in enumerate(lambdas):
		# ridge regression
		w_x, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
		rmse_tr.append(np.sqrt(2 * compute_mse(y_tr, tx_tr, w_x)))
		rmse_te.append(np.sqrt(2 * compute_mse(y_te, tx_te, w_x)))

		#print("proportion={p}, degree={d}, lambda={l:.15f}, Training RMSE={tr:.10f}, Testing RMSE={te:.10f}".format(
		#       p=ratio, d=degree, l=lambda_*10**15, tr=rmse_tr[ind], te=rmse_te[ind]))
	
	plt.semilogx(lambdas, rmse_tr, color='b', marker='*', label="Train error")
	plt.semilogx(lambdas, rmse_te, color='r', marker='*', label="Test error")
	plt.xlabel("lambda")
	plt.ylabel("RMSE")
	plt.title("Ridge regression for polynomial degree " + str(degree))
	leg = plt.legend(loc=1, shadow=True)
	leg.draw_frame(False)
	plt.show()



	#plot_train_test(rmse_tr, rmse_te, lambdas, degree)

	#rr_w, rr_loss = ridge_regression(y, tx, lambda_)

	#return rr_w, rr_loss, degree


def run_logistic_regression(y, x):

	y, tx = build_model_data(y,x) 
	
	initial_w = np.zeros(len(y))
	gamma = 0.1
	max_iters = 50


	w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)

	return lr_w, lr_loss


def main():
	yb_train, input_data_train, ids_train = load_csv_data('/Users/sigrid/Documents/Skole/Rolex/data/train.csv', sub_sample=False)
	yb_test, input_data_test, ids_test = load_csv_data('/Users/sigrid/Documents/Skole/Rolex/data/test.csv', sub_sample=False)
	#yb_train, input_data_train, ids_train = load_csv_data('/Users/maikenberthelsen/Documents/EPFL/Machine Learning/Project 1/Rolex/data/train.csv', sub_sample=False)
	#yb_test, input_data_test, ids_test = load_csv_data('/Users/maikenberthelsen/Documents/EPFL/Machine Learning/Project 1/Rolex/data/test.csv', sub_sample=False)
	#yb_train, input_data_train, ids_train = load_csv_data('/Users/idasandsbraaten/Dropbox/Rolex/data/train.csv', sub_sample=False)
	#yb_test, input_data_test, ids_test = load_csv_data('/Users/idasandsbraaten/Dropbox/Rolex/data/test.csv', sub_sample=False)


	x_train = standardize(input_data_train)

	x_test = standardize(input_data_test)
	y_test, tx_test = build_model_data(x_test,yb_test)


	#gd_w, gd_loss = run_gradient_descent(yb_train, x_train)

	#sgd_w, sgd_loss = run_stochastic_gradient_descent(yb_train, x_train)

	rr_w, rr_loss, degree = run_ridge_regression(yb_train,x_train)
	tx_test = build_poly(x_test,degree)

	#ls_w, ls_loss, degree = run_least_square(yb_train,x_train)
	#tx_test = build_poly(x_test,degree)

	#lr_w, lr_loss = run_logistic_regression(yb_train, x_train)

	#tune_ridge_regression(yb_train,x_train)


	#Make predictions
	y_pred = predict_labels(rr_w, tx_test)

	create_csv_submission(ids_test, y_pred, 'test7_rr') #lager prediction-fila i Rolex-mappa med det navnet

	return 0;


### Run main function
main()
