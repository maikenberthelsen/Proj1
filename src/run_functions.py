import numpy as np
from implementations import *
from helpers import *
import matplotlib.pyplot as plt


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
	lambda_ = 0.01
	degree = 6

	tx = build_poly(x,degree)

	rr_w, rr_loss = ridge_regression(y, tx, lambda_)

	return rr_w, rr_loss, degree



def run_logistic_regression(y, x):

	y, tx = build_model_data(x,y) 
	initial_w = np.zeros((tx.shape[1], 1))
	y = np.expand_dims(y, axis=1)
	
	gamma = 0.09
	max_iters = 10

	lr_w, lr_loss = logistic_regression(y, tx, initial_w, max_iters, gamma)

	return lr_w, lr_loss

def run_logistic_regression2(y, x):

	y, tx = build_model_data(x,y) 
	initial_w = np.zeros((tx.shape[1], 1))
	y = np.expand_dims(y, axis=1)
	

	# initial_w = [-0.3428, 0.01885391, -0.26018961, -0.22812764, -0.04019317, -0.00502791,
	# 	0.32302178, -0.01464156, 0.23543933, 0.00973278, -0.0048371, -0.13453445,
 #  		0.13354281, -0.0073677, 0.22358728, 0.01132979, -0.00372824, 0.25739398,
 #  		0.02175267,  0.01270975,  0.12343641, -0.00613063, -0.09086221, -0.20328519,
 #  		0.05932847, 0.049829, 0.05156299, -0.01579745, -0.00793358, -0.00886158, -0.10660545]

	gamma = 0.1
	max_iters = 10

	lr_w, lr_loss = logistic_regression2(y, tx, initial_w, max_iters, gamma)

	return lr_w, lr_loss

def run_logistic_regression3(y, x):

	y, tx = build_model_data(x,y) 
	initial_w = np.zeros((tx.shape[1], 1))
	y = np.expand_dims(y, axis=1)
	

	# initial_w = [-0.3428, 0.01885391, -0.26018961, -0.22812764, -0.04019317, -0.00502791,
	# 	0.32302178, -0.01464156, 0.23543933, 0.00973278, -0.0048371, -0.13453445,
 #  		0.13354281, -0.0073677, 0.22358728, 0.01132979, -0.00372824, 0.25739398,
 #  		0.02175267,  0.01270975,  0.12343641, -0.00613063, -0.09086221, -0.20328519,
 #  		0.05932847, 0.049829, 0.05156299, -0.01579745, -0.00793358, -0.00886158, -0.10660545]

	gamma = 0.01
	max_iters = 10

	lr_w, lr_loss = logistic_regression3(y, tx, initial_w, max_iters, gamma)


	return lr_w, lr_loss

def run_reg_logistic_regression(y, x):

	y, tx = build_model_data(x,y) 
	initial_w = np.zeros((tx.shape[1], 1))
	y = np.expand_dims(y, axis=1)
	

	# initial_w = [-0.3428, 0.01885391, -0.26018961, -0.22812764, -0.04019317, -0.00502791,
	# 	0.32302178, -0.01464156, 0.23543933, 0.00973278, -0.0048371, -0.13453445,
 #  		0.13354281, -0.0073677, 0.22358728, 0.01132979, -0.00372824, 0.25739398,
 #  		0.02175267,  0.01270975,  0.12343641, -0.00613063, -0.09086221, -0.20328519,
 #  		0.05932847, 0.049829, 0.05156299, -0.01579745, -0.00793358, -0.00886158, -0.10660545]

	gamma = 0.09
	lambda_ = 0.03
	max_iters = 10

	rlr_w, rlr_loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

	return rlr_w, rlr_loss





