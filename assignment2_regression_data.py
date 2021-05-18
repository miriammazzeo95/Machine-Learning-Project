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
from matplotlib.pyplot import figure, plot, title, xlabel, xticks, yticks, ylabel, show, legend
from matplotlib.pyplot import boxplot, subplot, hist, ylim
from sklearn import tree
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sklearn.linear_model as lm

plt.close('all')

df = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data', delim_whitespace=True, header=None, na_values=["?"])
df2 = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.test', delim_whitespace=True, header=None, na_values=["?"])

data = df.append(df2, ignore_index=True)
cnan=df.isna().sum()
[nrows,ncol]=df.shape

#columns=list(data)
#print(cnan)
cols=list()
for i in range(0,len(cnan)):
    #print(cnan[i])
    if i!=0 and i!=2 and i!=6 and i!=7 and i!=8 and i!=9 and i!=11 and i!=22 and i<24:
        if cnan[i]/nrows<.25:
            cols.append(i)
            #print(i)
            #print(i+1)
            #print(cnan[i]/nrows)
        

df1=df[df.columns[cols]]
df1_1=df[df.columns[3:5]]
df1_2=df[[0,22]]
rnan3=df1_2.isna().sum(axis=1)
rnan2=df1_1.isna().sum(axis=1)
rnan=df1.isna().sum(axis=1)
rows=list()
ncol = len(cols)
for x in range (0,len(rnan)):
    if rnan2[x]>1 or rnan[x]/ncol >.25:
        #rows.append(x)
        df1.drop(x, inplace = True) 
        
        
#data=df1[df1[rows]]
        
print(df1.isna().sum().sum())
count=0
data = np.array(df1.get_values(), dtype=np.float64)
missing_idx = df1.isna()
mn = df1.mean(axis=0)
md = df1.mode(axis=0,numeric_only=True)
for y in df1.columns:
    #print(y)
    mdy = md[y];
    #print(mdy[0])
    if y!=3 and y!=4 and y!=5 and y!=18 and y!=19:
        mdy1=mdy[0]
        df1.loc[:, y] = df1.loc[:, y].fillna(mdy1)
    else:
        mny=mn[y]
        df1.loc[:, y] = df1.loc[:, y].fillna(mny)
    
  
raw_data = df1.get_values()
X = raw_data[:, :]

attributeNames = [x+1 for x in cols]

classLabels = raw_data[:,-1] 

classNames = np.unique(classLabels)

classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

N = len(y)
M = len(attributeNames)
C = len(classNames)

X_c = X.copy();
y_c = y.copy();

attributeNames_c = attributeNames.copy();

var = {'Surgery Treated', 'Age', 'Temperature', 'Pulse', 'Respiratory Rate', 'Pain', 'Abdominal Distention', 'Volume Blood', 'Proteins', 'Outcome', 'Surgical Lesion'};
#varplot= { 'Age', 'Temperature', 'Pulse', 'Respiratory Rate', 'Pain', 'Abdominal Distention', 'Volume Blood', 'Proteins', 'Surgical Lesion'};
varplot= np.array([ 'Age', 'Temperature', 'Pulse', 'Respiratory Rate', 'Pain', 'Abdominal Distention', 'Volume Blood', 'Proteins', 'Surgical Lesion']);
varplot2= np.array([ 'Age', 'Temperature', 'Pulse', 'Respiratory Rate', 'Pain1', 'Pain2','Pain3','Pain4','Pain5', 'D1', 'AD2', 'AD3', 'AD4', 'Volume Blood', 'Proteins', 'Surgical Lesion']);

#age attribute with values between 0 and 1
for i in np.arange(len(X[:,0])):
    if X[i,0]==9:
        X[i,0]='1';
    else:
        X[i,0]='0';
print(X[:,0])
        
#pain attribute with values between 0 and 1
X[:,4]=X[:,4]-1
X[:,5]=X[:,5]-1
X[:,8]=X[:,8]-1



# Since the 'age'(0), 'Pain'(4), 'abdominal distention'(5) and 'surgical lesion'(8) class information (which is now the last column in X_r) is a
# categorical variable, we will do a one-out-of-K encoding of the variable:
# The encoded information is now a 255x2 matrix. This corresponds to 150
# observations, and 2 possible ages. For each observation, the matrix
# has a row, and each row has one 0 and a single 1. The placement of the 1
# specifies which age the observations was.

pain = np.array(X[:, 4], dtype=int).T
K_pain = 5 
pain_encoding = np.zeros((pain.size, K_pain))
pain_encoding[np.arange(pain.size), pain] = 1

dist = np.array(X[:, 5], dtype=int).T
K_dist = 4 
dist_encoding = np.zeros((dist.size, K_dist))
dist_encoding[np.arange(dist.size), dist] = 1

X_old=X
# We need to replace the last column in X (which was the not encoded
# version of the species data) with the encoded version:
X = np.concatenate( (X_old[:,0:4],pain_encoding,dist_encoding, X_old[:,6:9]), axis=1)
#

#STANDARDIZATION
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(0)
U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T
N,M = X.shape

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

# Compute variance explained by principal components
#rho = (S*S) / (S*S).sum()
#threshold = 0.95
###############################################################################
#statistic: instograms
attributeNames = varplot2[0:16]
N = len(y)
M = len(attributeNames)
C = len(classNames)


###REGRESSION ON PULSE
# Split dataset into features and target vector
pulse_idx = 2
y = X[:,pulse_idx]

X_cols = list(range(0,pulse_idx)) + list(range(pulse_idx+1,len(attributeNames)))
x = Y2[:,X_cols]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(x,y)

# Predict alcohol content
y_est = model.predict(x)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('pulse (actual)'); ylabel('pulse (estimated)');
subplot(2,1,2)
hist(residual,40)
xlabel('difference between estimated and actual')
ylabel('number of occurances')

show()


###REGRESSION ON RESPIRATION RATE
# Split dataset into features and target vector
rate_idx = 3
yr = X[:,rate_idx]

X_cols_r = list(range(0,rate_idx)) + list(range(rate_idx+1,len(attributeNames)))
xr = Y2[:,X_cols_r]

# Fit ordinary least squares regression model
model.fit(xr,yr)

# Predict alcohol content
y_est_r = model.predict(xr)
residual_r = y_est_r-yr

# Display scatter plot
figure()
subplot(2,1,1)
plot(yr, y_est_r, '.')
xlabel('respiration rate (actual)'); ylabel('respiration rate (estimated)');
subplot(2,1,2)
hist(residual_r,40)
xlabel('difference between estimated and actual')
ylabel('number of occurances')

show()









############################################
print("you are awesome")
