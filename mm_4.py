# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:24:31 2020

@author: miriam
"""
import numpy as np
import pandas as pd
from mm_1 import *
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from matplotlib.pyplot import figure, subplot, hist, xlabel, xticks, ylim, show, boxplot, bar, legend
from scipy.stats import zscore



# eliminate Region attribute
X=X_stat.copy()
cols=cols_stat
#attribute names being analysed
varplot= [cols[0],'Health','Freedom','Generosity','Corruption','Pop. Density','Coastline',
          'Net Migration','Infant Mortality','GDP per capita','Literacy','Phones','Arable',
          'Crops','Climate 1','Climate 2','Climate 3','Climate 4','Climate 5','Climate 6', 
          *cols[20:25], 'Score']

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)


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



###HISTIOGRAM
#figure()
#u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
#for i in range(M):
#    subplot(u,v,i+1)
#    hist(X[:,i])
#    xlabel(attributeNames[i])
#    ylim(0,N/2)
#    
#show()

###############################################################################

X=X - np.ones((N,1))*X.mean(0)
# We start with a box plot of each attribute
figure()
title('Countries: Boxplot')
boxplot(X)
xticks(range(1,M+1), attributeNames, rotation=45)
# From this there may be some outliers in the GDP
# attribute but not sure
# However, it is impossible to see the distribution of the data, because
# the axis is dominated by these extreme outliers. To avoid this, we plot a
# box plot of standardized data (using the zscore function).
figure()
title('Countries: Boxplot (standarized)')
boxplot(zscore(X, ddof=1), attributeNames)
xticks(range(1,M+1), attributeNames, rotation=45)





X=X_n
figure()
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c) # binary mask to extract elements of class c
#    class_mask = np.nonzero(y==c)[0].tolist()[0] # indices of class c
    
    boxplot(X[class_mask,:])
    #title('Class: {0}'.format(classNames[c]))
    title('Class: '+classNames[c])
#    xlabel(varplot)
    xticks(range(1,len(varplot)+1), [a[:7] for a in varplot], rotation=90)
    y_up = X.max()+(X.max()-X.min())*0.1; y_down = X.min()-(X.max()-X.min())*0.1
    ylim(y_down, y_up)

show()



X=X1
figure()
title('Attribute average value by class')
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for c in range(C):
    for i in range(M):
        subplot(u,v,i+1)
        class_mask = (y==c) # binary mask to extract elements of class c
    #    class_mask = np.nonzero(y==c)[0].tolist()[0] # indices of class c
        
        bar(classNames[c], X[class_mask,i].mean())


show()





