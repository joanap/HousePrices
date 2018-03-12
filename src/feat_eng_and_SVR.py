# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:02:38 2017

@author: Alexandre

This code uses the kaggle framework found in an independent repository.
To use it for the first time folow these quick instructions:
	clone this repo "https://github.com/AlexGomesDS/Kaggle_project_framework" 
    and add the folder /src to sys.path, for example if you cloned to 
	  C:\GIT\Kaggle_project_framework
	run:
		import sys
		sys.path.append('C:\\GIT\\Kaggle_project_framework\\src')
	Now you can extend the class AbstractPreprocessor from anywhere like this:
		from features.preprocessor.abstract_preprocessor import AbstractPreprocessor
"""

#%% 0.1 Import libraries
import pandas as pd
import numpy as np

from framework_extensions.HousePricePred_mean_price_label_encoder import Mean_Price_Preprocessor
from framework_extensions.HousePricePred_mean_mode_preprocessor import Mean_Mode_Preprocessor

from sklearn import svm
import matplotlib.pyplot as plt
#%% 0.2 Import datasets
train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='Id')
test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='Id')

#%% 1.0 Preprocess the data
data_preprocessor = Mean_Price_Preprocessor('SalePrice')

# datasets after preprocessing and feature engineering
data_preprocessor.prepare(train_dataframe)
X_train, y_train    = data_preprocessor.cook_and_split(train_dataframe)

X_test, _           = data_preprocessor.cook_and_split(test_dataframe)


#%% feature engineer
from sklearn.model_selection import cross_val_score

def calc_adj_r2(x, y, N = 5):
    model = svm.SVR(C=1e3)
    #model = svm.SVR(kernel='poly')
    r2 = cross_val_score(model, x, y.values.ravel(), cv=N)
    n = len(x)/N
    k = len(x.columns)
    return (1 - (1-r2)*(n-1)/(n-k-1)).mean()

def select_feats(dfclean, y):
    vfeats = []
    res = []
    for feat in dfclean:
        l_r2 = []
        l_feats = []        
        for feat2 in dfclean:
            if feat2 not in vfeats:
                l_feats.append(feat2)
                l_r2.append(calc_adj_r2(dfclean[vfeats + [feat2]], y, N=2))
                
        df_lr2 = pd.DataFrame(l_r2, index=l_feats)
        R2_max = df_lr2[0].max()
        arg_R2_max = df_lr2[0].argmax()
        
        if len(res)>1 and R2_max < res[-1]:
            break
            
        vfeats.append(arg_R2_max)
        res.append(R2_max)
        print("{} features. Max adj R2 of {}. New feat added was {}".format
              (len(vfeats), 
               R2_max,
               arg_R2_max
               ))
    
    return pd.DataFrame(res, columns=['R2'], index = vfeats)
    
res = select_feats(X_train, y_train)
#%%
plt.plot(pd.DataFrame(res.R2.values))#.R2.values)

X_train_eng = X_train[res.index.values]
X_test_eng = X_test[res.index.values]

model = svm.SVR()
model.fit(X_train_eng, y_train)
pred = model.predict(X_test_eng)

y_pred = pd.DataFrame(np.exp(pred), 
                      index = X_test.index, 
                      columns = ['SalePrice'])

y_pred.to_csv('..\\Submissions\\SupportVectorRegressor_feat_eng.csv')