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

X=X_stat.copy()
cols=cols_stat


# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

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
X_std = preprocessing.scale(X)



###############################################################################
# Subtract mean value from data
Y = X_n

f=figure()
r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,0))
plt.xticks(r, varplot, rotation=90)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('attribute standard deviations')
show()




#
### Investigate how standardization affects PCA
#############################################################################
## Try this *later* (for last), and explain the effect
##X_s = X.copy() # Make a to be "scaled" version of X
##X_s[:, 2] = 100*X_s[:, 2] # Scale/multiply attribute C with a factor 100
## Use X_s instead of X to in the script below to see the difference.
## Does it affect the two columns in the plot equally?
#
#
## Subtract the mean from the data
#Y1 = X - np.ones((N, 1))*X.mean(0)
#
## Subtract the mean from the data and divide by the attribute standard
## deviation to obtain a standardized dataset:
#Y2 = X - np.ones((N, 1))*X.mean(0)
#Y2 = Y2*(1/np.std(Y2,0))
## Here were utilizing the broadcasting of a row vector to fit the dimensions 
## of Y2
#
## Store the two in a cell, so we can just loop over them:
#Ys = [Y1, Y2]
#titles = ['Zero-mean', 'Zero-mean and unit variance']
#threshold = 0.9
## Choose two PCs to plot (the projection)
#i = 0
#j = 1
#
#
#
## Make the plot
#plt.figure(figsize=(10,15))
#plt.subplots_adjust(hspace=.4)
#plt.title('Effect of standardization')
#nrows=2
#ncols=2
#for k in range(2):
#    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
#    U,S,Vh = svd(Ys[k],full_matrices=False)
#    V=Vh.T # For the direction of V to fit the convention in the course we transpose
#    # For visualization purposes, we flip the directionality of the
#    # principal directions such that the directions match for Y1 and Y2.
#    if k==1: V = -V; U = -U; 
#    
#    # Compute variance explained
#    rho = (S*S) / (S*S).sum() 
#    
#    # Compute the projection onto the principal components
#    Z = U*S;
#    
#    # Plot projection
#    plt.subplot(nrows, ncols, 1+k)
#    C = len(classNames)
#    for c in range(C):
#        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
#    plt.xlabel('PC'+str(i+1))
#    plt.xlabel('PC'+str(j+1))
#    plt.title(titles[k] + '\n' + 'Projection' )
#    plt.legend({'no', 'yes'})
#    plt.axis('equal')
#    
#    
##################################################################            
#    # Plot cumulative variance explained
#    plt.subplot(nrows, ncols,  3+k);
#    plt.plot(range(1,len(rho)+1),rho,'x-')
#    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
#    plt.plot([1,len(rho)],[threshold, threshold],'k--')
#    plt.title('Variance explained by principal components');
#    plt.xlabel('Principal component');
#    plt.ylabel('Variance explained');
#    plt.legend(['Individual','Cumulative','Threshold'])
#    plt.grid()
#    plt.title(titles[k]+'\n'+'Variance explained')
#
#plt.show()

# How does this translate to the actual data and its projections?
# Looking at the data for water:

# Projection of water class onto the 2nd principal component.
#all_water_data = Y[y==4,:]

#print('First water observation')
#print(all_water_data[0,:])

# Based on the coefficients and the attribute values for the observation
# displayed, would you expect the projection onto PC2 to be positive or
# negative - why? Consider *both* the magnitude and sign of *both* the
# coefficient and the attribute!

# You can determine the projection by (remove comments):
#print('...and its projection onto PC2')
#print(all_water_data[0,:]@V[:,1])
# Try to explain why?

##############################################################################
##var2= {'Surgery Treated', 'Age','', 'Temperature', 'Pulse', 'Respiratory Rate','','','','', 'Pain','', 'Abdominal Distention','','','','','', 'Volume Blood', 'Proteins','','', 'Outcome', 'Surgical Lesion','','','',''};
#var2=np.array(['Surgery Treated', 'Age', 'Temperature', 'Pulse', 'Respiratory Rate', 'Pain', 'Abdominal Distention', 'Volume Blood', 'Proteins', 'Outcome', 'Surgical Lesion'])
#plt.figure()
#for a in range(V.shape[1]):
#        plt.arrow(0,0, V[a,i], V[a,j])
#        plt.text(V[a,i], V[a,j], varplot[a])
#plt.xlim([-1,1])
#plt.ylim([-1,1])
#plt.xlabel('PC'+str(i+1))
#plt.ylabel('PC'+str(j+1))
#plt.grid()
#    # Add a unit circle
#plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
#         np.sin(np.arange(0, 2*np.pi, 0.01)));
#plt.title(titles[k] +'\n'+'Attribute coefficients')
#plt.axis('equal')
#
#pcs = [0,1,2]
#f = figure()
#legendStrs = ['PC'+str(e+1) for e in pcs]
#c = ['r','g','b']
#bw = .2
#r = np.arange(1,M+1)
#for i in pcs:    
#    plt.bar(r+i*bw, V[:,i], width=bw)
#plt.xticks(r+bw, varplot, rotation=40)
#plt.xlabel('Attributes')
#plt.ylabel('Component coefficients')
#plt.legend(legendStrs)
#plt.grid()
#plt.title('PCA Component Coefficients')
#plt.show()
################################################################
 
#pca = PCA(n_components=2)
#pca.fit(X_c)
#pcacomp=pca.components_
#pcavar=pca.explained_variance_
#
#fig, ax = plt.subplots()
##fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
#
## plot principal components
#X_pca = pca.transform(X_c)
#ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
##draw_vector([0, 0], [0, 3], ax=ax[0])
##draw_vector([0, 0], [3, 0], ax=ax[0])
##ax[0].axis('equal')
#ax.set(xlabel='component 1', ylabel='component 2',
#          title='principal components')
#
#pca = PCA(n_components=3)
#pca.fit(X)
#X_pca = pca.transform(X)
#print("original shape:   ", X.shape)
#print("transformed shape:", X_pca.shape)
#
#X_new = pca.inverse_transform(X_pca)
#plt.figure()
#plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
#plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
#plt.axis('equal');
