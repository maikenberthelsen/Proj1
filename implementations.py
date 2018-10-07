#all implementation goes here

import numpy as np


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
	#dette er en test

    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return losses, ws



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    return

def least_squares(y, tx):
    return

def ridge_regression(y, tx, initial_w, max_iters, gamma):
    return

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma):
    return
