# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:24:31 2020

@author: miriam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mm_1 import *
from ann_validate import *

import seaborn as sns; sns.set()
from sklearn import preprocessing
from matplotlib.pyplot import figure, subplot, hist, xlabel, xticks, xlim, ylim, show, boxplot, bar, legend, clim
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
from scipy.stats import zscore
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from toolbox_02450 import feature_selector_lr, bmplot
from toolbox_02450 import rlr_validate

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

#########################################################
X=X_n
y=y_r


# Values of lambda
lambdas = np.power(10.,range(-5,9))
#lambdas = np.logspace(-10, -1, 50)
K=10
CV = model_selection.KFold(K, shuffle=True)


Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,len(lambdas)))
Error_test_rlr = np.empty((K,len(lambdas)))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_rlr = np.empty((M,K))
w_noreg = np.empty((M,K))
w = np.empty((M,K,len(lambdas)))

train_err_vs_lambda= np.empty((K,1))
test_err_vs_lambda =np.empty((K,1))
mean_w_vs_lambda=np.empty((K,1))
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
        Error_train_rlr[k,l] = np.power(y_train-X_train @ w[:,k,l].T,2).mean(axis=0)
        Error_test_rlr[k,l] = np.power(y_test-X_test @ w[:,k,l].T,2).mean(axis=0)
        
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
#    ### Compute squared error with all features selected (no feature selection)
#    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
#    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
#    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    k+=1

gen_err=np.empty((len(lambdas),1))
gen_err2=np.empty((len(lambdas),1))
for i in range(0,len(lambdas)):
    gen_err2[i] = Error_test_rlr[:,i].mean()
    gen_err[i] = Error_test_rlr[:,i].sum()*(12/127)

min_gen_err = gen_err.min()
optimum_lambda = lambdas[np.argmin(gen_err)]


train_err=np.empty((len(lambdas),1))
for i in range(0,len(lambdas)):
    train_err[i] = Error_train_rlr[:,i].mean()


figure( figsize=(12,8))   
title('Generalization error')
loglog(lambdas,gen_err, 'r.-', lambdas,train_err,'b')
xlabel('Regularization factor (λ)')
ylabel('Generalization error (crossvalidation)')
legend(['Generalization error','Train error'])
grid()
opt_model=lambdas[np.where(gen_err==gen_err.min())[0]]

# Fit ordinary least squares regression model
model = lm.Ridge(alpha=10)
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Happiness score (true)'); ylabel('HAppiness score (estimated)');
subplot(2,1,2)
hist(residual,40)

show()


#########################################################
X=X_n
y=y_r
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
opt_lambda= np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))



k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    
    
    opt_val_err, opt_lambda[k], mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda[k] * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda[k])))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
        
int('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))


###################################################################################

