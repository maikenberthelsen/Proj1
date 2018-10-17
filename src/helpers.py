# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1) #predictions
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int) #indexes
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def standardize(x):
    """Standardize the original data set."""
    x -= np.mean(x, axis=0)
    #print(x)
    x /= np.std(x, axis=0)

    return x

def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    N = len(y)
    #tx = np.c_[np.ones(N), x]
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    return y, tx

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse =  1/(2 * len(e)) * e.dot(e)
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0/(1 + np.exp(-t))
    #return np.exp(t)/(1 + np.exp(t))

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    ret = np.ones([len(x),1])
    for d in range (1,degree+1):
        # for hver dimensjon må det legges til en kolonne (som er x elementvist opphøyd i d)
        # dette kan gjøres med np.c_

        ret = np.c_[ret,np.power(x,d)]
    return ret

def split_data(y, x, ratio, seed=10):
    np.random.seed(seed)
    N = len(y)

    index = np.random.permutation(N)
    index_tr = index[: int(np.floor(N*ratio))]
    index_te = index[int(np.floor(N*ratio)) :]

    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]

    return x_tr, x_te, y_tr, y_te

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
    x_tr=x[tr_indices].flatten()

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    w = ridge_regression(y_tr, tx_tr, lambda_)

    loss_tr=np.sqrt(2*compute_mse(y_tr, tx_tr, w))
    loss_te=np.sqrt(2*compute_mse(y_te, tx_te, w))

    return loss_tr, loss_te

def cross_validation_demo():
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for lambda_ in lambdas:
        rmse_tr_temp= []
        rmse_te_temp= []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
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
