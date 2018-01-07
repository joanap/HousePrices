# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:02:38 2017

@author: Alexandre
"""

#%% 0.1 Import libraries
import pandas as pd

from features.preprocessor.mean_price_label_encoder import Mean_Price_Preprocessor
from features.preprocessor.mean_mode_preprocessor import Mean_Mode_Preprocessor

from sklearn.linear_model import ElasticNet

#%% 0.2 Import datasets
train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='Id')
test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='Id')

#%% 1.0 Preprocess the data
data_preprocessor1 = Mean_Mode_Preprocessor('SalePrice')
data_preprocessor = Mean_Price_Preprocessor('SalePrice')

# datasets after preprocessing and feature engineering
X_train, y_train = data_preprocessor.prepare_and_cook(train_dataframe)
X_test, _ = data_preprocessor.cook(test_dataframe)

#%% Elastic net

regr = ElasticNet()
regr.fit(X_train, y_train)
pred = regr.predict(X_test)
