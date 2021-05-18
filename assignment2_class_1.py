# -*- coding: utf-8 -*-
"""

@author: Miry
"""
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy import stats
from scipy.io import loadmat
from matplotlib.pyplot import figure, plot, title, colorbar, imshow, xlabel, xticks, yticks, ylabel, show, legend
from matplotlib.pyplot import boxplot, subplot, hist, ylim
from sklearn import tree
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call, rlr_validate
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from numpy import cov, median
from sklearn.preprocessing import StandardScaler

from assignment2_regression_data import X, Y2, attributeNames

def rlogistic_validate(X,y,lambdas,cvf):
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X,y):
        
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        for l in range(0,len(lambdas)):
       
            mdl = lm.logistic.LogisticRegression(penalty='l2', C=1/lambdas[l] )
            mdl=mdl.fit(X_train, y_train)
            
            y_train_est = mdl.predict(X_train)
            y_test_est = mdl.predict(X_test)    
            
            train_error[f,l] = np.sum(y_train_est != y_train) / len(y_train)
            test_error[f,l] = np.sum(y_test_est != y_test) / len(y_test) 
        f = f+1
    
    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    train_err_inn=train_error
    test_err_inn=test_error
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, train_err_inn, test_err_inn
#########################################################################################################

#Fit logistic regression model
y= X[:,15]
x_l=Y2[:, :15]
X=X[:, :15]
N,M=X.shape

K=10
#T = len(lambdas)
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

misclass_rate = np.empty((K,1))
output_lambda = np.empty((K,1))

CV = model_selection.KFold(K, shuffle=True)
# Values of lambda

lambdas = np.power(10.,range(-5,8))
#lambdas = np.logspace(-10,10 , 10)
acc = np.empty((K,1))

k=0
scaler = StandardScaler()
for train_index, test_index in CV.split(X,y):
    dy = []
    yhat= []
    y_true=[]
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    internal_cross_validation = 10    
    
   # print('internal cross validation')
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, train_err_inn, test_err_inn = rlogistic_validate(X_train, y_train, lambdas, internal_cross_validation)
    output_lambda[k]=opt_lambda
    # Standardize outer fold based on training set,
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = lm.logistic.LogisticRegression(penalty='l2', C=1/opt_lambda)
    model = model.fit(X_train,y_train)

    # Classify horses and assess probabilities
    y_est_l = model.predict(X_test)
    dy.append( y_est_l )
        # errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
    
    
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    
    yhat = np.concatenate(yhat)
    y_true = np.concatenate(y_true)
    yhat[:] # predictions made by first classifier.
    # Compute accuracy here.
    acc[k]= np.sum(yhat[:] != y_true) 
#    y_est_surg_prob = model.predict_proba(x_test)[:, 0] 

    # Evaluate classifier's misclassification rate over entire training data
    misclass_rate[k] = np.sum(y_est_l != y_test) / float(len(y_est_l))
    
    k+=1  
    
plt.figure()
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambdas, train_err_inn[9]*100)
plt.semilogx(lambdas, test_err_inn[9]*100)
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error'],loc='upper right')
#################################################################
acc_l=acc.mean()
y_true_l=y_true

#0Out[72]: 
#array([0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 1., 1., 0., 1., 1.,
#       0., 1., 0., 0., 0., 0., 1., 1.])
print("Good job!")
