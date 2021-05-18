# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:24:31 2020

@author: miriam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mm_1 import *

import seaborn as sns; sns.set()
from sklearn import preprocessing
from matplotlib.pyplot import figure, subplot, hist, xlabel, xticks, yticks, xlim, ylim, show, boxplot, bar, legend, clim
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
varplot= [cols[0],'Health','Freedom','Generosity','Corruption','Pop. Density','Coastline',
          'Net Migration','Infant Mortality','GDP per capita','Literacy','Phones','Arable',
          'Crops','Climate 1','Climate 2','Climate 3','Climate 4','Climate 5','Climate 6', 
          *cols[20:25], 'Score']

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


################################################
y=y_r
X=X_n
# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X_n,y)

# Predict alcohol content
y_est = model.predict(X_n)
residual = y_est-y




# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Happiness score (true)'); ylabel('Happiness score (estimated)');
subplot(2,1,2)
hist(residual,40)
xlim(-2,2)
show()


#######################################################
y = y_c

# Fit logistic regression model
model = lm.logistic.LogisticRegression()
model = model.fit(X,y)

# Classify wine as White/Red (0/1) and assess probabilities
y_est = model.predict(X)
y_est_happy_prob = model.predict_proba(X)[:, 0] 

# Define a new data object (new type of country), as in exercise 5.1.7
x = X[3,:].reshape(1,-1)
# Evaluate the probability of x being an happy country (class=0) 
x_class = model.predict_proba(x)[0,0]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nProbability of given sample being a happy country: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_happy_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_happy_prob[class1_ids], '.r')
xlabel('Data object (country sample)'); ylabel('Predicted prob. of class Happy');
legend(['happy', 'unhappy'])
ylim(-0.01,1.5)

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
K = 5
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
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

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
    lambdaI = opt_lambda * np.eye(M)
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
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

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


X=X_stat
X=preprocessing.normalize(X)

varplot= [cols[0],'Health','Freedom','Generosity','Corruption','Pop. Density','Coastline',
          'Net Migration','Infant Mortality','GDP per capita','Literacy','Phones','Arable',
          'Crops','Climate 1','Climate 2','Climate 3','Climate 4','Climate 5','Climate 6', 
          *cols[20:25], 'Score']

## Next we plot a number of atttributes
Attributes = [10,5,11,25]
NumAtr = len(Attributes)

figure(figsize=(12,12))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
            if m1==NumAtr-1:
                xlabel(varplot[Attributes[m2]])
            else:
                xticks([])
            if m2==0:
                ylabel(varplot[Attributes[m1]])
            else:
                yticks([])
#            ylim(0,X.max()*1.1)
#            xlim(0,X.max()*1.1)
legend(classNames)
show()

Attributes = [1,2,11,25]

M = len(Attributes)

figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,Attributes[m2]]), np.array(X[class_mask,Attributes[m2]]), '.')
            if m1==M-1:
                xlabel(varplot[Attributes[m2]])
            else:
                xticks([])
            if m2==0:
                ylabel(varplot[Attributes[m2]])
            else:
                yticks([])
            #ylim(0,X.max()*1.1)
            #xlim(0,X.max()*1.1)
legend(classNames)

show()

plot(np.array(X[class_mask,Attributes[m2]]), np.array(X[class_mask,Attributes[m2]]))










