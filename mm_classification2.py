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
from numpy import cov, median
from sklearn.preprocessing import StandardScaler
from toolbox_02450 import mcnemar

from mm_1 import *
from validate import *
from toolbox_02450 import *
import seaborn as sns; sns.set()
from scipy.stats import zscore
from sklearn import preprocessing

#########################################################################################################
def baseline( ytrain, ytest ):
    y_predict= ytest*0+np.median(ytrain)
    return y_predict
#####################
###############################################################################
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
    scaler = StandardScaler()
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
def knn_validate(X,y,lambdas,cvf):
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
    w = np.empty((M,cvf,len(lambdas)+1))
    train_error = np.empty((cvf,len(lambdas)+1))
    test_error = np.empty((cvf,len(lambdas)+1))
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
        
        for l in range(1,len(lambdas)+1):
       
            mdl = KNeighborsClassifier(n_neighbors=l)
            mdl.fit(X_train, y_train)
            
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
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, test_error
#########################################################################################################



# Extract vector y, convert to NumPy array
y_c = np.asarray([classDict[value] for value in classLabels])

#REGRESSION
y_r=df['Score'].values

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

# separate the data from the target attributes
varplot= attributeNames

# normalize the data attributes
#Normalization refers to rescaling real valued numeric attributes into the range 0 and 1.
#It is useful to scale the input attributes for a model that relies on 
#the magnitude of values, such as distance measures used in 
#k-nearest neighbors and in the preparation of coefficients in regression.
X_n = preprocessing.normalize(X)

# standardize the data attributes
#Standardization refers to shifting the distribution of each attribute to have 
#a mean of zero and a standard deviation of one (unit variance).
#X_std = preprocessing.scale(X)
X_std=zscore(X, ddof=1)

#############################################################################

X=X_std
y=y_c

K=10

misclass_rate_l = np.empty((K,1))
misclass_rate_knn = np.empty((K,1))
misclass_rate_bs= np.empty((K,1))
output_lambda = np.empty((K,1))

opt_l=np.empty((K,1))
opt_n=np.empty((K,1))

CV = model_selection.KFold(K, shuffle=False)
# Values of lambda
#lambdas = np.power(10.,range(-5,9))
L=40
knn= np.arange(1,L+1)
log= np.logspace(-10, -1, 50)
acc = np.empty((K,3))
misclass_rate = np.empty((K,3))

k=0
scaler = StandardScaler()
models=[0, 1, 2]
yhat= []
y_true=[]


for train_index, test_index in CV.split(X,y):
    dy = []
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    
    internal_cross_validation = 10  

    print('logistics innervalidation')
    lambdas=log
    #solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda, train_err_inn, test_err_inn = rlogistic_validate(X_train, y_train, lambdas, internal_cross_validation)
    opt_l[k]=opt_lambda
    model = lm.logistic.LogisticRegression(penalty='l2', C=1/opt_lambda, solver='lbfgs')
    model = model.fit(X_train,y_train)
    y_est_l = model.predict(X_test)
    dy.append( y_est_l )
    acc[k,0]= np.sum(y_est_l != y_true) 
     # Evaluate classifier's misclassification rate over entire training data
    misclass_rate[k,0] = np.sum(y_est_l != y_test) / float(len(y_est_l))
    
    print('baseline')
    y_est_bl = baseline(y_train, y_test)
    dy.append( y_est_bl )
    acc[k,1]= np.sum(y_est_bl != y_true) 
     # Evaluate classifier's misclassification rate over entire training data
    misclass_rate[k,1] = np.sum(y_est_bl != y_test) / float(len(y_est_bl))
     
    print('KNN')
    nn=knn
    opt_val_err, opt_nn, mean_w_vs_nn, train_err_vs_nn, test_err_vs_nn, test_error = knn_validate(X_train, y_train, nn, internal_cross_validation)
    opt_n[k]=opt_nn
    model = KNeighborsClassifier(n_neighbors=opt_nn);
    model=model.fit(X_train, y_train);
    y_est_knn = model.predict(X_test)
    dy.append( y_est_knn )
    acc[k,2]= np.sum(y_est_knn!= y_true) 
     # Evaluate classifier's misclassification rate over entire training data
    misclass_rate[k,2] = np.sum(y_est_knn != y_test) / float(len(y_est_knn))
    
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    
    k=k+1

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)
# Compute the Jeffreys interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)
print("misclass rate:", misclass_rate.mean(axis=0))

[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)
print("misclass rate:", misclass_rate.mean(axis=0))


[thetahat, CI, p] = mcnemar(y_true, yhat[:,2], yhat[:,1], alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)
print("misclass rate:", misclass_rate.mean(axis=0))









