import numpy as np
from implementations import *
from helpers import *
import matplotlib.pyplot as plt


def run_gradient_descent(y, x):

	y, tx = build_model_data(x,y)

	max_iters = 2000
	gamma = 0.1 #Ikke høyere enn 0.15, da konvergerer det ikke
	initial_w = np.zeros(tx.shape[1])
	
	#initial_w = [-0.3428, 0.01885391, -0.26018961, -0.22812764, -0.04019317, -0.00502791,
	#	0.32302178, -0.01464156, 0.23543933, 0.00973278, -0.0048371, -0.13453445,
  	#	0.13354281, -0.0073677, 0.22358728, 0.01132979, -0.00372824, 0.25739398,
  	#	0.02175267,  0.01270975,  0.12343641, -0.00613063, -0.09086221, -0.20328519,
  	#	0.05932847, 0.049829, 0.05156299, -0.01579745, -0.00793358, -0.00886158, -0.10660545]

	gd_w, gd_loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)

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
	degree = 12
	tx = build_poly(x,degree)

	ls_w, ls_loss = least_squares(y, tx)

	return ls_w, ls_loss, degree


def run_ridge_regression(y,x):
	lambda_ = 0.0001
	degree = 10

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

def run_logistic_regression_hessian(y, x):
	y, tx = build_model_data(x,y) 
	initial_w = np.zeros((tx.shape[1], 1))
	y = np.expand_dims(y, axis=1)

	gamma = 0.01
	max_iters = 10

	lr_w, lr_loss = logistic_regression_hessian(y, tx, initial_w, max_iters, gamma)


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

	gamma = 0.0004
	lambda_ = 0.0001
	max_iters = 100

	rlr_w, rlr_loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

	return rlr_w, rlr_loss

#def bagging_ridge_regression(y,x):
#	nb_bags = 10
#	lambda_ = 0.0001
#	degree = 10
#	tx = build_poly(x,degree)
#	for i in range(nb_bags):  
        #choosing a random subsample(with replacement) of the data with size equal to half of the sample size
        #a= x[np.random.choice(x.shape[0],x.shape[0]//2, replace=True)]
        
 #       a = tx[np.random.randint(x.shape[0], size=x.shape[0]//2), :]
#		rr_w, rr_loss = ridge_regression(y, tx, lambda_)


#	y_pred = predict_labels(rr_w, tx_test)

#	return rr_w, rr_loss, degree

'''def gradient_descent_model(train):
	y = train[0]
	x = train[1::]
	tx = np.c_[np.ones((y.shape[0], 1)), x]
	max_iters = 20
	gamma = 0.1 #Ikke høyere enn 0.15, da konvergerer det ikke
	initial_w = np.zeros(tx.shape[1])
	gd_w, gd_loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
	#y_pred = predict_labels(gd_w, tx_test)
	return gd_w

def least_square_model(train):
	y = train[0]
	x = train[1::]
	degree = 12
	tx = build_poly(x,degree)
	ls_w, ls_loss = least_squares(y, tx)
	return ls_w

def ridge_regression_model(train):
	y = train[0]
	x = train[1::]
	degree = 10
	tx = build_poly(x,degree)
	rr_w, rr_loss = ridge_regression(y, tx, lambda_)
	return rr_w

def logistic_regression_model(train):
	y = train[0]
	x = train[1::]
	degree = 1
	tx = build_poly(x,degree)
	initial_w = np.zeros((tx.shape[1], 1))
	y = np.expand_dims(y, axis=1)
	gamma = 0.01
	max_iters = 10
	lr_w, lr_loss = logistic_regression3(y, tx, initial_w, max_iters, gamma)
	return lr_w

def gradient_descent_predict():
	max_iters = 2000
	gamma = 0.1 #Ikke høyere enn 0.15, da konvergerer det ikke
	initial_w = np.zeros(tx.shape[1])
	gd_w, gd_loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
	y_pred = predict_labels(gd_w, tx_test)

'''
def build_poly_vec(vec,degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    ret = np.ones(1)
    for d in range (1,degree+1):
        # for hver dimensjon må det legges til en kolonne (som er x elementvist opphøyd i d)
        # dette kan gjøres med np.c_
        #np.append(ret,np.power(vec,d))#, axis =1)
        ret = np.concatenate((ret,np.power(vec,d)))#,axis =1)
    return ret

def gradient_descent_model(train):
	y = train[:,[0]]
	y = np.squeeze(np.asarray(y))
	x = np.delete(train,0,axis=1)
	tx = np.c_[np.ones((y.shape[0], 1)), x]
	max_iters = 20
	gamma = 0.1 #Ikke høyere enn 0.15, da konvergerer det ikke
	initial_w = np.zeros(tx.shape[1])
	gd_w, gd_loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)
	return gd_w

def least_square_model(train):
	y = train[:,[0]]
	y = np.squeeze(np.asarray(y))
	x = np.delete(train,0,axis=1)
	degree = 12
	tx = build_poly(x,degree)
	ls_w, ls_loss = least_squares(y, tx)
	return ls_w

def ridge_regression_model(train):
	y = train[:,[0]]
	y = np.squeeze(np.asarray(y))
	x = np.delete(train,0,axis=1)
	degree = 10
	lambda_ = 0.0001
	tx = build_poly(x,degree)
	rr_w, rr_loss = ridge_regression(y, tx, lambda_)
	return rr_w

def logistic_regression_model(train):
	y = [i[-1] for i in train]
	y = np.squeeze(np.asarray(y))
	x = np.delete(train,0,axis=1)
	degree = 1
	tx = build_poly(x,degree)
	initial_w = np.zeros((tx.shape[1], 1))
	y = np.expand_dims(y, axis=1)
	gamma = 0.01
	max_iters = 10
	lr_w, lr_loss = logistic_regression3(y, tx, initial_w, max_iters, gamma)
	return lr_w

def gradient_descent_predict(w, row):
	print("row wo delete", row)
	row = np.delete(row,0)
	print("row deleted", row)

	np.insert(row,0,1)
	print("rowinserted", row)
	new_row = np.transpose(row)
	y_pred = predict_labels_row(w, new_row)
	return y_pred

def least_square_predict(w,row):
	degree = 12
	new_row = np.delete(row,1)
	new_row = np.transpose(new_row)
	t_row = build_poly_vec(new_row,degree)
	y_pred = predict_labels_row(w, t_row)
	return y_pred

def ridge_regression_predict(w,row):
	new_row = np.delete(row,1)
	degree = 10
	new_row = np.transpose(new_row)
	t_row = build_poly_vec(new_row,degree)
	y_pred = predict_labels_row(w, t_row)
	return y_pred

def logistic_regression_predict(w,row):
	#y = train[:,[0]]
	degree = 1
	new_row = np.delete(row,1)
	new_row = np.transpose(new_row)
	t_row = build_poly_vec(new_row,degree)
	y_pred = predict_labels_row(w, t_row)
	return y_pred


# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
	stacked_row = list()
	for i in range(len(models)):
		prediction = predict_list[i](models[i], row)
		stacked_row.append(prediction)
	#stacked_row.append(row[-1])
	stacked_row.append(row[0])
	#new_row = row[1:len(row)]
	#print(new_row, new_row.shape, stacked_row, len(stacked_row))
	#for i in range(0, len(stacked_row)):
	#	np.append(new_row,stacked_row[i])
	rowlist = row[1:len(row)-1].tolist()
	return rowlist + stacked_row#new_row#row[0:len(row)-1] + stacked_row

def stacking(y,x,yte,xte):
	train = np.c_[y.T,x]
	test = np.c_[yte.T,xte]
	model_list = [gradient_descent_model, least_square_model, ridge_regression_model]#[least_square_model]#[gradient_descent_model, least_square_model, ridge_regression_model]
	predict_list = [gradient_descent_predict, least_square_predict,ridge_regression_predict] #[least_square_predict]#[gradient_descent_predict, least_square_predict, ridge_regression_predict]
	models = list()
	for i in range(len(model_list)):
		model = model_list[i](train)
		models.append(model)
	stacked_dataset = list()
	print("finished with models")
	for row in train:
		#i = 1
		stacked_row = to_stacked_row(models, predict_list, row)
		stacked_dataset.append(stacked_row)
		#np.append(stacked_dataset,stacked_row,axis = 0)
		#i = i +1
		#print("finished with row:" , i)
	#print(stacked_dataset)
	stacked_model = logistic_regression_model(stacked_dataset)
	print("finished training")
	predictions = list()
	for row in test:
		#j = 1
		stacked_row = to_stacked_row(models, predict_list, row)
		#np.append(stacked_dataset,stacked_row,axis = 0)
		stacked_dataset.append(stacked_row)
		#print("model", stacked_model)
		#print("row", stacked_row)
		prediction = logistic_regression_predict(stacked_model, stacked_row)
		#prediction = round(prediction)
		#if prediction > 0.5:
		#	prediction = 1
		#else:
		#	prediction = -1
		predictions.append(prediction)
	print("finished test")
	return predictions




