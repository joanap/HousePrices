# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")

# Creating the matrix of features ( matrix of independent variables)
X = dataset.iloc[:,:-1].values

# Matrix of dependent variables
Y = dataset.iloc[:,-1].values

# fill missing data with the mean value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder_X = LabelEncoder()
X[:,0] = labelenconder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

# Splitting the dataset into training and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)