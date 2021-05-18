from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
import sklearn.linear_model as lm

from assignment2_regression_data import X, x, y, X_cols, pulse_idx, attributeNames

X = X[:, X_cols]
#X =x
name = attributeNames[X_cols]

attributeNames = name[:]
N, M = X.shape

# Values of lambda
lambdas = np.power(10.,range(-5,9))
K=10
CV = model_selection.KFold(K, shuffle=True)


Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
train_error_rlr = np.empty((K,len(lambdas)))
test_error_rlr = np.empty((K,len(lambdas)))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
w = np.empty((M,K,len(lambdas)))
train_err_vs_lambda= np.empty((K,1))
test_err_vs_lambda =np.empty((K,1))
mean_w_vs_lambda=np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
N_k=np.empty((K,1))

k=0   
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    N_k[k]=X_test.shape[0]
    
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model 
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    
    for l in range(0,len(lambdas)):
 
        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = lambdas[l]* np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w[:,k,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error_rlr[k,l] = np.power(y_train-X_train @ w[:,k,l].T,2).mean(axis=0)
        test_error_rlr[k,l] = np.power(y_test-X_test @ w[:,k,l].T,2).mean(axis=0)
        
#    opt_val_err = np.min(np.mean(test_error,axis=0))
#    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
#    train_err_vs_lambda = np.mean(train_error,axis=0)
#    test_err_vs_lambda = np.mean(test_error,axis=0)
#    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    
    k+=1

gen_err=np.empty((len(lambdas),1))
for i in range(0,len(lambdas)):
    gen_err[i] = test_error_rlr[:,i].mean()

figure( figsize=(12,8))   
title('Generalization error')
loglog(lambdas,gen_err,'b.-')
xlabel('Regularization factor (λ)')
ylabel('Generalization error (crossvalidation)')
grid()
opt_model=lambdas[np.where(gen_err==gen_err.min())[0]]

print('Ran program')      
#    
#    
## Display results
#print('Linear regression without feature selection:')
#print('- Training error: {0}'.format(Error_train.mean()))
#print('- Test error:     {0}'.format(Error_test.mean()))
#print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
#print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
#print('Regularized linear regression:')
#print('- Training error: {0}'.format(Error_train_rlr.mean()))
#print('- Test error:     {0}'.format(Error_test_rlr.mean()))
#print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
#print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))
#
#print('Weights in last fold:')
#for m in range(M):
#    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
#
#
#figure()
#title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
#loglog(output_lambda,Error_train_rlr,'b.-',output_lambda,Error_test_rlr,'r.-')
#xlabel('Regularization factor (lambda)')
#ylabel('Squared error (crossvalidation)')
#legend(['Train error','Validation error'])
#grid()
#
#






