
from implementations import *
from helpers import *

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

    acc = float(np.sum(y_te == y_pred))/len(y_te)

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
    

def degree_selection(y,x):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-3,0, 5)
    degrees = range(2,6+1)

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    accvectors = []
    
    for d_ind, lambda_ in enumerate(lambdas):
        print(d_ind)

        # define lists to store the loss of training data and test data
        rmse_tr = []
        rmse_te = []
        acc = []

        # cross validation
        for ind, degree in enumerate(degrees):
            loss_tr = []
            loss_te = []
            acc_1 = []
            
            for k in range (k_fold):
                temp_loss_tr, temp_loss_te, temp_acc = cross_validation(y, x, k_indices, k, lambda_, degree)
                loss_tr.append(np.sqrt(2*temp_loss_tr))
                loss_te.append(np.sqrt(2*temp_loss_te))
                acc_1.append(temp_acc)

            
            rmse_tr.append(np.mean(loss_tr))
            rmse_te.append(np.mean(loss_te))
            acc.append(np.mean(acc_1))

        
        if d_ind == 0:
            accvectors = acc
        else:
            accvectors = np.vstack((accvectors, acc))
    
    #cross_validation_visualization_degree(degrees, rmse_tr, rmse_te)
    plot_accs_degs(degrees, lambdas, accvectors)


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


def plot_acc_degs(degs, acc):
    plt.plot(degs, acc, marker=".", color='b', label='Accuracy')
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("cross validation degree accuracy")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()

def plot_accs_degs(degs, lambdas, accs):
    
    for i in range(len(lambdas)):
        label = 'lambda =' + str(lambdas[i])
        plt.plot(degs, accs[i,:], marker=".", label=label)
    
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("cross validation degree accuracy")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()