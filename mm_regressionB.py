# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:24:31 2020

@author: miriam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mm_1 import *
from validate import *
from toolbox_02450 import *

import seaborn as sns; sns.set()
from sklearn import preprocessing
from matplotlib.pyplot import figure, subplot, hist, xlabel, xticks, xlim, ylim, show, boxplot, bar, legend, clim
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
from scipy.stats import zscore
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from scipy import stats
from sklearn.neural_network import MLPRegressor
import scipy.stats as st

# Extract vector y, convert to NumPy array
y_c = np.asarray([classDict[value] for value in classLabels])

#REGRESSION
y_r = df['Score'].values

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
#X_n = preprocessing.normalize(X)
X_n = stats.zscore(X)


# standardize the data attributes
#Standardization refers to shifting the distribution of each attribute to have 
#a mean of zero and a standard deviation of one (unit variance).
#X_std = preprocessing.scale(X)
X_std=zscore(X, ddof=1)


#######################################################################
    #Fit regression models
#########################################################
X=X_n.copy()
y=y_r.copy()
## Add offset attribute
#X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
#M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
innerK=10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas_power = np.power(10.,range(-5,9))
lambdas_log= np.logspace(-10, -1, 50)
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
misclass_rate_l = np.empty((K,1))
misclass_rate_knn = np.empty((K,1))
misclass_rate_bs= np.empty((K,1))
Ei_ann=np.empty((K,1))
Ei_rlr=np.empty((K,1))
Ei_bl=np.empty((K,1))
opt_l=np.empty((K,1))
opt_h=np.empty((K,1))
yhat = []
y_true = []
r = []
loss=2
k=0
for train_index, test_index in CV.split(X,y):
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
#    X_train = scaler.fit_transform(X_train)
#    X_test = scaler.transform(X_test)

     
    dy = []
    print('regularised linear regression innervalidation')
    lambdas=lambdas_log
    opt_val_err_ridge, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X,y,lambdas,cvf=innerK)
    opt_l[k]=opt_lambda
    model = lm.Ridge(alpha=opt_lambda)
    model = model.fit(X_train, y_train)
    y_est_l = model.predict(X_test)
    dy.append( y_est_l )
    yhatA = model.predict(X_test)
    Ei_rlr[k] = np.power(y_test-y_est_l,2).mean(axis=0)
    
    print('baseline')
    y_est_bl = baseline(y_train, y_test)
    dy.append( y_est_bl )
    Ei_bl[k] = np.power(y_test-y_est_bl,2).mean(axis=0)
    
    print('ANN')
    hu=17
    errors_ann, opt_val_err_ann, opt_hiddenunits = ann_validate(X_train, y_train, innerK, hu)
    model = MLPRegressor(solver='sgd', hidden_layer_sizes=(opt_hiddenunits,), activation="tanh",
                       max_iter=10000, random_state=1)
    opt_h[k]=opt_hiddenunits
    model=model.fit(X_train, y_train)
    y_est_ann = model.predict(X_test)
    dy.append( y_est_ann )
    yhatB = model.predict(X_test)[:, np.newaxis] 
    Ei_ann[k] = np.power(y_test-y_est_ann,2).mean(axis=0)
    k=k+1
    
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    y_true.append(y_test)
    r.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ) )
        
yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true)

# Compute accuracy 
Egen_rlr= np.abs(y_true-yhat[:,0])**2
Egen_bl=  np.abs(y_true-yhat[:,1])**2
Egen_ann=  np.abs(y_true-yhat[:,2])**2

E_rlr= np.mean(Egen_rlr)
E_bl= np.mean(Egen_bl)
E_ann= np.mean(Egen_ann)

#OR COMPUTE GENERALIZATION ERROR as in Algorithm 6
Eg_rlr= (len(y_test)/len(y))*np.sum(Ei_rlr)
Eg_bl= (len(y_test)/len(y))*np.sum(Ei_bl)
Eg_ann= (len(y_test)/len(y))*np.sum(Ei_ann)


# Initialize parameters and run test appropriate for setup I
# perform statistical comparison of the models
# compute z with squared error.
zA = Egen_rlr
# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zA = Egen_rlr
zB = Egen_ann
z = zA - zB
CI_1 = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p_1 = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print(CI_1)
print(p_1)

###############################################################
zA = Egen_rlr
zB = Egen_bl
z = zA - zB
CI_1 = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p_1 = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print(CI_1)
print(p_1)

###############################################################
zA = Egen_ann
zB = Egen_bl
z = zA - zB
CI_1 = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p_1 = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print(CI_1)
print(p_1)


###############################################################################
# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K
p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)
print('setupII:')
print(p_setupII)
print(CI_setupII)