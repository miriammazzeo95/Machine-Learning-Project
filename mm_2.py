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
from scipy.stats import zscore

# separate the data from the target attributes
# eliminate Region attribute
X=X_stat.copy()
cols=cols_stat

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

attributeNames=cols
# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

#attribute names being analysed
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
X_std = preprocessing.scale(X)
#X_std=zscore(X, ddof=1)


###############################################################################
# Subtract mean value from data
Y=X_std.copy()
#Y = X- np.ones((N,1))*X.mean(0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.9
# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#more than 90% of the variation in the data is explained
#by the first 3 principal components

###############################################################################
Y=X_n.copy()

#Y = X - np.ones((N,1))*X.mean(0)
# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    
# Project the centered data onto principal component space
Z = Y @ V
# Indices of the principal components to be plotted
i = 0
j =3
# Plot PCA12 of the data
plt.figure()
plt.title('PCA1_4')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
show()


i = 0
j =2
# Plot PCA13 of the data
plt.figure()
plt.title('PCA1-3')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
show()


U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T
N,M = X.shape
## We saw in 2.1.3 that the first 3 components explaiend more than 90
## percent of the variance. Let's look at their coefficients:
pcs = [0,1]
f = figure()
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['darksalmon','b','g']
bw = .3
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i],color = c[i], width=bw)
plt.xticks(r+bw, varplot, rotation=90)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()

###################################################################################
## Inspecting the plot, we see that the 2nd principal component has large
## (in magnitude) coefficients for attributes A, E and H. We can confirm
## this by looking at it's numerical values directly, too:
#print('PC2:')
#print(V[:,1].T)
#
f=figure()
r = np.arange(1,X_n.shape[1]+1)
plt.bar(r, np.std(X_n,0))
plt.xticks(r, varplot, rotation=90)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('attribute standard deviations')
show()

f=figure()
r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, varplot, rotation=90)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('attribute standard deviations')
show()

a=np.arange(1,27,1)
i=0;
j=1;
 # Plot attribute coefficients in principal component space
f=figure()
for att in range(V.shape[1]):
     plt.arrow(0,0, V[att,i], V[att,j])
     plt.text(V[att,i], V[att,j], varplot[att])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1))
plt.ylabel('PC'+str(j+1))
plt.grid()
    # Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title('Attribute coefficients')
plt.axis('equal')
show() 



### Investigate how standardization affects PCA
#############################################   ################################
# Try this *later* (for last), and explain the effect
#X_s = X.copy() # Make a to be "scaled" version of X
#X_s[:, 2] = 100*X_s[:, 2] # Scale/multiply attribute C with a factor 100
# Use X_s instead of X to in the script below to see the difference.
# Does it affect the two columns in the plot equally?

X = X_stat.copy()
# Subtract the mean from the data
Y1 = X - np.ones((N, 1))*X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
#Y2 = X - np.ones((N, 1))*X.mean(0)
#Y2 = Y2*(1/np.std(Y2,0))
Y2 = X_std.copy()
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.6
# Choose two PCs to plot (the projection)
i = 0
j = 1



# Make the plot
plt.figure(figsize=(11,6))
plt.subplots_adjust(hspace=.4)
plt.title('Effect of standardization')
nrows=2
ncols=2
for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;
    
    # Plot projection
    plt.subplot(nrows, ncols, 1+k)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.xlabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(classNames, loc='upper right')
    plt.axis('equal')
    
    
#################################################################            
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  3+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'], loc='upper right')
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')

plt.show()

## Subtract mean value from data
#Y = X_n
#
#f=figure()
#r = np.arange(1,X.shape[1]+1)
#plt.bar(r, np.std(X,0))
#plt.xticks(r, varplot, rotation=90)
#plt.ylabel('Standard deviation')
#plt.xlabel('Attributes')
#plt.title('attribute standard deviations')
#show()
