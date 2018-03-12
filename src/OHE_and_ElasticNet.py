# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:02:38 2017

@author: Alexandre
"""

#%% 0.1 Import libraries
import pandas as pd
import numpy as np

from framework_extensions.HousePricePred_mean_price_label_encoder import Mean_Price_Preprocessor
from framework_extensions.HousePricePred_mean_mode_preprocessor import Mean_Mode_Preprocessor

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
#%% 0.2 Import datasets
train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='Id')
test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='Id')

#%% 1.0 Preprocess the data
data_preprocessor = Mean_Price_Preprocessor('SalePrice')

cat_train = train_dataframe.select_dtypes(include=[np.dtype('O')])
cat_test = test_dataframe.select_dtypes(include=[np.dtype('O')])

cat_le = {}

for i in cat_train:
    cat_le[i] = LabelEncoder().fit(pd.concat([cat_train[i],cat_test[i]]).fillna('NaN'))
    cat_train[i] = cat_le[i].transform(cat_train[i].fillna('NaN'))
    cat_test[i] = cat_le[i].transform(cat_test[i].fillna('NaN'))

enc = OneHotEncoder()
enc.fit(pd.concat([cat_test,cat_train]))
X_ohe = pd.DataFrame(enc.transform(cat_train).toarray(),index=train_dataframe.index)
Xt_ohe =pd.DataFrame(enc.transform(cat_test).toarray(),index=test_dataframe.index)
# datasets after preprocessing and feature engineering
data_preprocessor.prepare(train_dataframe)
X_train, y_train    = data_preprocessor.cook_and_split(train_dataframe)
X_train2 = pd.concat([X_train, X_ohe],axis=1)
X_test, _           = data_preprocessor.cook_and_split(test_dataframe)
X_test2 = pd.concat([X_test, Xt_ohe],axis=1)

#%% Elastic net

regr = ElasticNetCV(normalize=True)
regr.fit(X_train2, y_train)
print(regr.alpha_)
pred = regr.predict(X_test2)

y_pred = pd.DataFrame(np.exp(pred), index = X_test.index, columns = ['SalePrice'])
y_pred.to_csv('..\\Submissions\\elastic_net_2.csv')

