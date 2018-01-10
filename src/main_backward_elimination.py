# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:02:38 2017

@author: Alexandre
"""

#%% 0.1 Import libraries
import pandas as pd
import numpy as np

from features.preprocessor.mean_price_label_encoder import Mean_Price_Preprocessor
from features.preprocessor.mean_mode_preprocessor import Mean_Mode_Preprocessor

#%% 0.2 Import datasets
train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='Id')
test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='Id')

#%% 1.0 Preprocess the data
data_preprocessor1 = Mean_Mode_Preprocessor('SalePrice')
data_preprocessor = Mean_Price_Preprocessor('SalePrice')

# datasets after preprocessing and feature engineering
data_preprocessor.prepare(train_dataframe)
X_train, y_train    = data_preprocessor.cook_and_split(train_dataframe)

X_test, _           = data_preprocessor.cook_and_split(test_dataframe)

#%% Method of backward elimination with the p-value:
# train the linear model, if there's at least one param with p-value 
# over the signf. level of 0.05 remove the top one and repeat

import statsmodels.formula.api as sm
import statsmodels.api as sm2
X_train2 = sm2.add_constant(X_train)
max_pv = 1
while max_pv > 0.05:
    smols = sm.OLS(y_train,X_train2)
    model = smols.fit()#_regularized(L1_wt=regr.alpha_)
    max_pv = model.pvalues.max()
    col_max_pv = model.pvalues.argmax()
    if max_pv > 0.05:
        X_train2 = X_train2.drop(col_max_pv, axis = 1)
        print("Removing column {} with P-value {}".format(col_max_pv, max_pv))

smols = sm.OLS(y_train,X_train2)
model = smols.fit()
model.summary()

#%%
X_test2 = sm2.add_constant(X_test)[X_train2.columns]
pred = model.predict(X_test2)

y_pred = pd.DataFrame(np.exp(pred), index = X_test.index, columns = ['SalePrice'])
y_pred.to_csv('..\\Submissions\\Backward_Elimination.csv')
#score:     0.134