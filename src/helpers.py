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

def predict_labels_row(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    if y_pred <= 0:
        y_pred = -1
    else:
        y_pred = 1
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

###############KOK###########################
def calculate_hessian(y, tx, w, pred):
    """return the hessian of the loss function."""
    #pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)

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


def remove999(x_train, pred_train, ids_train, x_test, ids_test): ## må sortere y også!!!

    x_train = np.concatenate((ids_train[:,None], x_train), axis=1)
    x_test = np.concatenate((ids_test[:,None], x_test), axis=1)  

    #Fix train data
    sig = x_train[pred_train == 1,:];
    back = x_train[pred_train == -1,:];

    for i in range(1,x_train.shape[1]):
        sig_mean = sum(sig[sig[:,i] != -999, i])/ len(sig[sig[:,i] != -999,i]);
        back_mean = np.sum(back[back[:,i] != -999,i])/ len(back[back[:,i] != -999,i]);
        test_mean = np.sum(x_test[x_test[:,i] != -999, i])/len(x_test[x_test[:,i] != -999,i]);
        train_mean = np.sum(x_train[x_train[:,i] != -999, i])/len(x_train[x_train[:,i] != -999,i]);


        #sig[sig[:,i] == -999,i] = sig_mean;
        #back[back[:,i] == -999,i] = back_mean;
        sig[sig[:,i] == -999,i] = train_mean;
        back[back[:,i] == -999,i] = train_mean;
        x_test[x_test[:,i] == -999,i] = test_mean; 

    
    x_train_fixed = np.vstack((sig,back))
    x_train_fixed = x_train_fixed[x_train_fixed[:,0].argsort(),]
    

    x_train_fixed = x_train_fixed[:,1:]
    x_test_fixed = x_test[:,1:]

    return x_train_fixed, x_test_fixed



def removecols(input_data_train, input_data_test, cols):

    input_data_train = np.delete(input_data_train,cols, axis = 1)
    input_data_test = np.delete(input_data_test,cols, axis = 1)
    

    return input_data_train, input_data_test
    

def logpositive(x_train, x_test):
    for i in range(1,x_train.shape[1]):
        if (np.all(x_train[:,i]) > 0 and np.all(x_test[:,i] > 0)):
            x_train[:,i] = np.log10(x_train[:,i])
            x_test[:,i] = np.log10(x_test[:,i])

    return x_train, x_test

def bootstrap_data(x, num_subSamp):
    temp_mean = np.zeros((1,x.shape[1]))
    temp_std = np.zeros((1,x.shape[1]))
    for i in range(num_subSamp):  
        #choosing a random subsample(with replacement) of the data with size equal to half of the sample size
        #a= x[np.random.choice(x.shape[0],x.shape[0]//2, replace=True)]
        a = x[np.random.randint(x.shape[0], size=x.shape[0]//2), :]
        temp_mean += np.mean(a, axis=0)
        temp_std += np.std(a, axis=0)
    bootstrapMean = temp_mean/num_subSamp
    bootstrapStd = temp_std/num_subSamp
    return bootstrapMean, bootstrapStd

def standardize_with_bootstrapping(x,num_subSamp):
    """Standardize the original data set."""
    b_mean, b_std = bootstrap_data(x, num_subSamp)
    x -= b_mean
    #print(x)
    x /= b_std

    return x


