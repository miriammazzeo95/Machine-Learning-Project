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


#Fit logistic regression model
y= X[:,15]
x_l=Y2[:, :15]
X=X[:, :15]
N,M=X.shape
#varplot2= np.array([ 'Age', 'Temperature', 'Pulse', 'Respiratory Rate', 'Pain1', 'Pain2','Pain3','Pain4','Pain5', 'D1', 'AD2', 'AD3', 'AD4', 'Volume Blood', 'Proteins', 'Surgical Lesion']);

K=10

opt_lambda=1e-4
# Fit logistic regression model
model = lm.logistic.LogisticRegression(C=1/opt_lambda)
model = model.fit(X,y)

# Classify horse and assess probabilities
y_est = model.predict(X)
y_est_surg_prob = model.predict_proba(X)[:, 0] 

# Define a new data object (new horse), as in exercise 5.1.7
x = np.array([1,	37,	80,	30,	0,	0,	0,	0,	1,	0,	0,	0,	1,	43,	8.4]).reshape(1,-1)
# Evaluate the probability of x needing surgery (class=0) 
x_class = model.predict_proba(x)[0,0]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nProbability of given sample needing surgery: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_surg_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_surg_prob[class1_ids], '.r')
xlabel('Data object (wine sample)'); ylabel('Predicted prob. of class White');
legend(['yes', 'no'])
ylim(-0.01,1.5)

show()

print('Ran Exercise 5.2.6')