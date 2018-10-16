import numpy as np
from implementations import *
from helpers import *
import matplotlib.pyplot as plt
from run_functions import *


def main():
	yb_train, input_data_train, ids_train = load_csv_data('/Users/sigrid/Documents/Skole/Rolex/data/train.csv', sub_sample=True)
	yb_test, input_data_test, ids_test = load_csv_data('/Users/sigrid/Documents/Skole/Rolex/data/test.csv', sub_sample=True)
	#yb_train, input_data_train, ids_train = load_csv_data('/Users/maikenberthelsen/Documents/EPFL/Machine Learning/Project 1/Rolex/data/train.csv', sub_sample=False)
	#yb_test, input_data_test, ids_test = load_csv_data('/Users/maikenberthelsen/Documents/EPFL/Machine Learning/Project 1/Rolex/data/test.csv', sub_sample=False)
	#yb_train, input_data_train, ids_train = load_csv_data('/Users/idasandsbraaten/Dropbox/Rolex/data/train.csv', sub_sample=False)
	#yb_test, input_data_test, ids_test = load_csv_data('/Users/idasandsbraaten/Dropbox/Rolex/data/test.csv', sub_sample=False)


	x_train = standardize(input_data_train)

	x_test = standardize(input_data_test)
	y_test, tx_test = build_model_data(x_test,yb_test)


	#gd_w, gd_loss = run_gradient_descent(yb_train, x_train)

	#sgd_w, sgd_loss = run_stochastic_gradient_descent(yb_train, x_train)

	#rr_w, rr_loss, degree = run_ridge_regression(yb_train,x_train)
	#tx_test = build_poly(x_test,degree)

	#ls_w, ls_loss, degree = run_least_square(yb_train,x_train)
	#tx_test = build_poly(x_test,degree)

	lr_w, lr_loss = run_logistic_regression(yb_train, x_train)

	#tune_ridge_regression(yb_train,x_train)


	#Make predictions
	y_pred = predict_labels(lr_w, tx_test)

	create_csv_submission(ids_test, y_pred, 'test8_lr') #lager prediction-fila i Rolex-mappa med det navnet

	return 0;


### Run main function
main()