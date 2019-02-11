# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 12:47:03 2019

@author: aotitoola
"""


#Data Preprocessing

#Importing the libraries

import numpy as np                   #contains math tools to include any type of math
import matplotlib.pyplot as plt      #help us plot nice charts
import pandas as pd                  #help to import and manage datasets


#Importing the dataset
dataset = pd.read_csv('Data.csv')

#create Matrix of features
X = dataset.iloc[:, :-1].values

#create dependent-variable vector
y = dataset.iloc[:, 3].values


#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)


#fit imputer object to matrix/feature x

'''
we are not fitting the imputer to all the matrix but to the column 
where we have missing data?
'''

imputer = imputer.fit(X[:, 1:3])
 
#replace missing data of matrix X by mean of column
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# ML Equations will think the encoding indicates some hierarchy and as such give it an order
# We create dummy variables to solve this problem, We use OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)










