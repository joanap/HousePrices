# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 20:49:16 2017

@author: ASSG
"""

# In[1]:
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
%matplotlib qt5
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model as lm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv('train.csv')
dftest = pd.read_csv("test.csv")

#%% boxplot of categorical features
data = df[["OverallQual","SalePrice"]]
sns.boxplot(x=df["OverallQual"],y=df["SalePrice"])

sns.boxplot(x=df["HeatingQC"],y=df["SalePrice"])
sns.boxplot(x=df["Neighborhood"],y=df["SalePrice"])

#%% linear regression with one_hot_encoded categorical features

lr = lm.LinearRegression()
X = df[["OverallQual","GrLivArea"]]
X = pd.concat([df.OverallQual,
                    df.GrLivArea,
                    df.GrLivArea**2],    axis = 1)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
y = df.SalePrice
xpred = lr.fit(X, y)

print("R2 = " + str(lr.score(X,y))) #R2 = 0.76

#%%

Xt = pd.concat([dftest.OverallQual,
                    dftest.GrLivArea,
                    dftest.GrLivArea**2],    axis = 1)
    
onehotencoder = OneHotEncoder(categorical_features=[0])

Xt = onehotencoder.fit_transform(Xt).toarray()

pred2 = xpred.predict(Xt)

dfpred2 = pd.concat([dftest.Id, 
                     pd.DataFrame(pred2, columns = ["SalePrice"])],
                    axis = 1)
dfpred2.astype(int).to_csv("submission_onehot.csv",index = False)

print("This submission yielded a score of 0.20251 which corresponded to 1413th place as of 27/09")