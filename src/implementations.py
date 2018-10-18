#all implementation goes here

import numpy as np
import datetime
from helpers import *
import matplotlib.pyplot as plt


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
	#threshold = 1e-8

	for n_iter in range(max_iters):
		# compute prediction, loss, gradient
		# tx should maybe not be transposed
		# not transposed when using large X
		#print(tx.dot(w))
		print(n_iter)
		
		prediction = sigmoid(tx.dot(w))
		
		loss = -(y.T.dot(np.log(prediction)) + (1-y).T.dot(np.log(1-prediction)))# + ((lambda_/2)*(np.linalg.norm(w)**2))
		gradient = tx.T.dot(prediction - y)

		# gradient w by descent update
		w = w - (gamma * gradient)
		
		# store w and loss
		ws.append(w)
		losses.append(loss)

		#if (len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold):
		#	break

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




################################################

def sigmoid2(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss_lr(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid2(tx.dot(w))
    print(pred)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid2(tx.dot(w))
    #print(pred.shape)
    grad = tx.T.dot(pred - y)
    return grad


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """

    # print(y.shape)
    # print(tx.shape)
    # print(w.shape)
    

    loss = calculate_loss_lr(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    #print(grad.shape)
    #print(w.shape)

    w -= gamma * grad
    return loss, w


def logistic_regression2(y, tx, initial_w, max_iters, gamma):
    # init parameters

    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        print(iter)
        
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    #print("loss={l}".format(l=calculate_loss_lr(y, tx, w)))

    return w, loss



# tentative suggestions for logistic regression
def logistic_regression3(y, tx, initial_w, max_iters, gamma):

    ws = [initial_w]
    losses = []
    w = initial_w
    #threshold = 1e-8

    for n_iter in range(max_iters):
        # compute prediction, loss, gradient
        # tx should maybe not be transposed
        # not transposed when using large X
        #print(tx.dot(w))
        print(n_iter)
        loss = 0
        for i in range(len(y)):
            loss = loss + (np.logaddexp(0, tx[i,:].T.dot(w)) - y[i]*tx[i].T.dot(w))
        
        prediction = sigmoid(tx.dot(w))
        
        #loss = -(y.T.dot(np.log(prediction)) + (1-y).T.dot(np.log(1-prediction)))# + ((lambda_/2)*(np.linalg.norm(w)**2))
        gradient = tx.T.dot(prediction - y)

        # gradient w by descent update
        w = w - (gamma * gradient)
        
        # store w and loss
        ws.append(w)
        losses.append(loss)

        #if (len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold):
        #   break

    #finds best parameters
    min_ind = np.argmin(losses)
    loss = losses[min_ind]
    w = ws[min_ind][:]
    
    return w, loss

########################

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    y_te=y[k_indices[k,:]]
    x_te=x[k_indices[k,:]]

    tr_indices=np.delete(k_indices, (k), axis=0)
    
    y_tr=y[tr_indices].flatten()
    x_tr = x[tr_indices].reshape(x.shape[0]-x_te.shape[0],x.shape[1])

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    w, loss = ridge_regression(y_tr, tx_tr, lambda_)


    loss_tr=np.sqrt(2*compute_mse(y_tr, tx_tr, w))
    loss_te=np.sqrt(2*compute_mse(y_te, tx_te, w))


    y_pred = predict_labels(w, tx_te)

    acc = 

    return loss_tr, loss_te, acc

def cross_validation_demo(y, x):
    seed = 5
    degree = 4
    k_fold = 10
    lambdas = np.logspace(-3, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for lambda_ in lambdas:
        rmse_tr_temp= []
        rmse_te_temp= []
        for k in range(k_fold):
            loss_tr, loss_te, acc = cross_validation(y, x, k_indices, k, lambda_, degree)
            rmse_tr_temp.append(loss_tr)
            rmse_te_temp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_temp))
        rmse_te.append(np.mean(rmse_te_temp))

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    
def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    plt.show()


def degree_selection(y,x):
    
    seed = 1
    #degree = 7
    k_fold = 4
    lambda_ = 0.01
    degrees = range(2,6+1)

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    acc = []
    
    # cross validation
    for ind, degree in enumerate(degrees):
        loss_tr = []
        loss_te = []
        acc_temp = []
        
        for k in range (k_fold):
            temp_loss_tr, temp_loss_te, temp_acc = cross_validation(y, x, k_indices, k, lambda_, degree)
            loss_tr.append(np.sqrt(2*temp_loss_tr))
            loss_te.append(np.sqrt(2*temp_loss_te))
        
        rmse_tr.append(np.mean(loss_tr))
        rmse_te.append(np.mean(loss_te))
    
    cross_validation_visualization_degree(degrees, rmse_tr, rmse_te)


def cross_validation_visualization_degree(degs, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.plot(degs, mse_tr, marker=".", color='b', label='train error')
    plt.plot(degs, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("degree")
    plt.ylabel("rmse")
    plt.title("cross validation degree")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()

