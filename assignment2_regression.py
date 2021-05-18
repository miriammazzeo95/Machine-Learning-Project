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
import numpy as np
from sklearn import tree
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from numpy import cov
from toolbox_02450 import rlr_validate
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
    
            mdl.fit(X_train, y_train)
            
            y_train_est = mdl.predict(X_train)
            y_test_est = mdl.predict(X_test)    
            
            train_error[f,l] = np.sum(y_train_est != y_train) / len(y_train)
            test_error[f,l] = np.sum(y_test_est != y_test) / len(y_test) 
        f = f+1
    

        w_est = mdl.coef_[0] 
#        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
#
#    min_error = np.min(test_error_rate)
#    opt_lambda_idx = np.argmin(test_error_rate)
#    opt_lambda = lambda_interval[opt_lambda_idx]

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda
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
#lambdas = np.power(10.,range(-5,9))
lambdas = np.logspace(-10, -1, 50)

k=0
scaler = StandardScaler()
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    internal_cross_validation = 10    
    
    print('internal cross validation')
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlogistic_validate(X_train, y_train, lambdas, internal_cross_validation)
    output_lambda[k]=opt_lambda
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = lm.logistic.LogisticRegression(C=1/opt_lambda)
    model = model.fit(X_train,y_train)

    # Classify horses and assess probabilities
    y_est_l = model.predict(X_test)
#    y_est_surg_prob = model.predict_proba(x_test)[:, 0] 


    # Evaluate classifier's misclassification rate over entire training data
    misclass_rate[k] = np.sum(y_est_l != y) / float(len(y_est_l))
    #
    ## Display classification results
    #print('\nProbability of given sample needing surgery: {0:.4f}'.format(x_class))
    #print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))
#
#    f = figure();
#    class0_ids = np.nonzero(y_l==0)[0].tolist()
#    plot(class0_ids, y_est_surg_prob[class0_ids], '.y')
#    class1_ids = np.nonzero(y_l==1)[0].tolist()
#    plot(class1_ids, y_est_surg_prob[class1_ids], '.r')
#    xlabel('Data object (horse sample)'); ylabel('Predicted prob. of surg');
#    legend(['yes', 'no'])
#    ylim(-0.01,1.5)
#
#    show()
    

    k+=1    

#################################################################
## Maximum number of neighbors
#
#y_knn=y_l
#N = len(y_knn)
#L=40
#x_knn= Y2[:,2:4]
#CV = model_selection.LeaveOneOut()
#errors = np.zeros((N,L))
#i=0
#
#for train_index, test_index in CV.split(x_knn, y_knn):
#    
#    # extract training and test set for current CV fold
#    X_train = x_knn[train_index,:]
#    y_train = y_knn[train_index]
#    X_test = x_knn[test_index,:]
#    y_test = y_knn[test_index]
#
#    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
#    for l in range(1,L+1):
#        knclassifier = KNeighborsClassifier(n_neighbors=l);
#        knclassifier.fit(X_train, y_train);
#        y_est = knclassifier.predict(X_test);
#        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
#
#    i+=1
#    
## Plot the classification error rate
#figure()
#plot(100*sum(errors,0)/N)
#xlabel('Number of neighbors')
#ylabel('Classification error rate (%)')
#show()
#
#print('Ran Exercise 6.3.2')

#
#
#C = len(np.array([ 'yes', 'no']))
#
#
## K-nearest neighbors
#K=30
#
## Distance metric (corresponds to 2nd norm, euclidean distance).
## You can set dist=1 to obtain manhattan distance (cityblock distance).
#dist=2
#metric = 'minkowski'
#metric_params = {} # no parameters needed for minkowski
#
## You can set the metric argument to 'cosine' to determine the cosine distance
##metric = 'cosine' 
##metric_params = {} # no parameters needed for cosine
#
## To use a mahalonobis distance, we need to input the covariance matrix, too:
##metric='mahalanobis'
##metric_params={'V': cov(X_train, rowvar=False)}
#
## Fit classifier and classify the test points
#knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
#                                    metric=metric,
#                                    metric_params=metric_params)
#knclassifier.fit(X_train, y_train)
#y_est = knclassifier.predict(X_test)
#
#
## Plot the classfication results
#figure()
#styles = ['ob', 'or']
#for c in range(1):
#    class_mask = (y_est==c)
#    plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
#    plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
#title('Synthetic data classification - KNN');
#
#
#print('Ran Exercise 6.3.1')











############################################
print("you are awesome")
