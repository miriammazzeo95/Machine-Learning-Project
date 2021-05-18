# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:12:17 2020

@author: miria
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:49:21 2019

@author: Zoey
"""
#import datacompy
import pandas as pd
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy import stats
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from categoric2numeric import *
#from sklearn import preprocessing
#from sklearn.preprocessing import OneHotEncoder

plt.close('all')

df1 = pd.read_csv('countries of the world.csv', sep=",", header=0, index_col=False)
df1_raw=df1.copy()
df2 = pd.read_csv('2019.csv', sep=",", header=0, index_col=False)
df2_raw=df2.copy()
df3 = pd.read_csv('country_profile_variables.txt', sep=";", header=0, index_col=False, encoding= 'unicode_escape')
# sorting data frame by name 
df1=df1.rename(columns={df1.columns[0]: 'Country'})
df1=df1.rename(columns={df1.columns[1]: 'Region'})

df2=df2.rename(columns={df2.columns[1]: 'Country'})

df3=df3.rename(columns={df3.columns[0]: 'Country'})


#To remove white space everywhere:
df1['Country'] = df1['Country'].str.replace(' ', '')
cols1 = list(df1.columns.values)
df1=  df1.replace({',': '.'}, regex=True)
df1=pd.concat([df1[cols1[0:2]],df1[cols1[2:]].astype(float)], axis=1, sort=False)
df2['Country'] = df2['Country'].str.replace(' ', '')
df3['Country'] = df3['Country'].str.replace(' ', '')
df1['Region'] = df1['Region'].str.replace(' ', '')
df1['Country'] = df1['Country'].str.replace('&', 'and')
df2['Country'] = df2['Country'].str.replace('&', 'and')
df1=df1.replace(to_replace ="Korea,North", value ="NorthKorea")
df1=df1.replace(to_replace ="Congo,Repub.ofthe", value ="Congo(Kinshasa)")
df1=df1.replace(to_replace ="Congo,Dem.Rep.", value ="Congo(Brazzaville)")
df1=df1.replace(to_replace ="Korea,South", value ="SouthKorea")
df1=df1.replace(to_replace ="Korea,North", value ="NorthKorea")
df1=df1.replace(to_replace ="CentralAfricanRep.", value ="CentralAfricanRepublic")

df2['Country'] = df2['Country'].str.replace('SouthSudan', 'Sudan')
df2['Country'] = df2['Country'].str.replace('NorthMacedonia', 'Macedonia')

df1.sort_values('Country', axis = 0, ascending = True, 
                 inplace = True, na_position ='last')
df2.sort_values("Country", axis = 0, ascending = True, 
                 inplace = True, na_position ='last')  

df12=pd.merge(df2, df1,on ='Country', how='inner')
df12.sort_values("Country") 

df=df12.copy()
cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Country')) #Remove b from list
cols.pop(cols.index('Region')) #Remove x from list

df = df[['Country','Region']+cols] #Create new dataframe with columns in the order you want
# Program to visualize missing values in dataset 
# Visualize missing values as a matrix 
msno.matrix(df)
df = df.dropna()

msno.matrix(df)

print(df.groupby('Region').Region.count())
#Region
#ASIA(EX.NEAREAST)    21    1
#BALTICS               2    2
#C.W.OFIND.STATES     10    3
#EASTERNEUROPE         6    4
#LATINAMER.&CARIB     21    5
#NEAREAST             11    6
#NORTHERNAFRICA        3    7
#NORTHERNAMERICA       1    8
#OCEANIA               2    9
#SUB-SAHARANAFRICA    37    10
#WESTERNEUROPE        16    11
#Name: Region, dtype: int64


mapping = {'ASIA(EX.NEAREAST)': 1, 'BALTICS': 2, 'C.W.OFIND.STATES':3, 'EASTERNEUROPE':4, 'LATINAMER.&CARIB':5, 
           'NEAREAST': 6, 'NORTHERNAFRICA':7, 'NORTHERNAMERICA':8, 'OCEANIA':9, 'SUB-SAHARANAFRICA':10,'WESTERNEUROPE':11}
df=df.applymap(lambda s: mapping.get(s) if s in mapping else s)

#CREATE CLASS
c= pd.Series((df['Score']>5)*1)
df.insert(28, "Class", c, True)

df['Class']=df['Class'].replace(to_replace=1, value='happy', regex=True)
df['Class']=df['Class'].replace(to_replace=0, value='unhappy', regex=True)

#SET INDEX
df=df.set_index('Country')
dfcols = list(df.columns.values) 

#population: change  unit to million
df['Population']=df['Population']/1000000
##Area: change  unit to Km
#df[dfcols[10]]=df[dfcols[10]]/1000000

#QUICK LOOK FOR OUTLIERS
#df.describe(include='all')
#fixing density to unit : ppl/m2
df['Pop. Density (per sq. mi.)']=df['Population']/df['Area (sq. mi.)']*1000000
#fixing GDP to unit : K$ per capita
df['GDP ($ per capita)']=df['GDP ($ per capita)']/1000
#df=df.rename(columns={df.columns[11]: 'Pop.Density'})
dataframe = pd.concat([df[dfcols[0]], df[dfcols[4:9]],df[dfcols[11:20]], df[dfcols[21:27]], df[dfcols[27]]], axis=1, sort=False)
dataframecols=[dfcols[0], *dfcols[4:9],*dfcols[11:20],*dfcols[21:27], dfcols[27]]
Xdf=dataframe[dataframecols[0:21]]
Xdf=Xdf.values

#enc = OneHotEncoder(handle_unknown='ignore')
#categoric2numeric converts data matrix with categorical columns given by
#numeric or text values to numeric columns using one out of K coding.
x0,xl0=categoric2numeric(Xdf[:,0])
x15,xl15=categoric2numeric(Xdf[:,15])
Xcod=np.concatenate([x0,Xdf[:,1:15],x15, Xdf[:,16:21]], axis=1)
######CONVERT LIST OF STR TO ARRAY OF FLOATS
#xl=[float(i) for i in xl]
#xl=np.array(xl)
xcodcols=[*xl0, *dataframecols[1:15], *xl15, *dataframecols[16:21]]
X=Xcod[:,11:36]
X1=X.copy()
cols=xcodcols[11:36]


attributeNames=cols
classLabels = list(df['Class'] )
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[cl] for cl in classLabels])

# Extract vector y, convert to NumPy array
y_c = np.asarray([classDict[value] for value in classLabels])

#REGRESSION
y_r = df['Score'].values

X_stat = np.hstack((X, np.expand_dims(y_r, axis=1)))
cols_stat = cols + ['Score']

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)


plt.close('all')


#SAVE DATAFRAME
#df.to_csv('df_1604.csv', index = False)

