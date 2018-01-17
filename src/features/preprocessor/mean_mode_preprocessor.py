# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:10:13 2017

@author: Alexandre


This class implements the following preprocessing methods:
    - Replace NaN values by their mean / most common label
    - Label encoding of the categorical features
"""


#%% Import onyl the necessary libraries
from features.preprocessor.abstract_preprocessor import AbstractPreprocessor 

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#%% Implementing class with our one version of the preprocessing methods

class Mean_Mode_Preprocessor(AbstractPreprocessor):

    # replace missing labels with the most common one (only when the NaNs are less than 10% of the full dataset)
    def _set_missing_cat_replacements(self, col):
        if col.count() / col.shape[0] > 0.90:
            return col.value_counts().index[0]
        return 'NaN'
    
    # set of dependent variables is only the last one (the one added in feat. eng.)
    def get_y(self, df):
        return df.loc[:, self.get_cols_to_predict()[-1]]
    
    # For categorical features, replace nan with the string "NaN" and apply labelencoding 
    # (also add the value Nan in case it appears in the test set and not in the train)
    def _set_cat_col_encoder(self, df):
        self.cat_encoders = {}
        for col_name in self.categorical_features:
            col_with_NaN = df[col_name].fillna('NaN').append(pd.DataFrame(['NaN']))
            self.cat_encoders[col_name] = LabelEncoder().fit(col_with_NaN.values.ravel())
    
    # Create a column with the log of the saleprice
    def _feat_eng_train(self, dataframe):
        cols_to_apply_log = list(self.get_cols_to_predict())
        
        for col_name in cols_to_apply_log:
            dataframe["Log_" + col_name] = np.log(dataframe[col_name])
            self.append_cols_to_predict("Log_" + col_name)


#%% Testing

if __name__ == '__main__':
    data_preprocessor = Mean_Mode_Preprocessor(["SalePrice"])

    train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='Id')
    test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='Id')

    data_preprocessor.prepare(train_dataframe)
    X_train, y_train = data_preprocessor.cook_and_split(train_dataframe)
    data_preprocessor.prepare(train_dataframe)
    X_train, y_train = data_preprocessor.cook_and_split(train_dataframe)
    X_test, _ = data_preprocessor.cook_and_split(test_dataframe)
    
    #tests
    #test if there are any NaNs
    assert(X_train.isnull().sum().sum() == 0)
    assert(X_test.isnull().sum().sum() == 0)
    
    #test the outputs have the correct shapes
    assert(X_train.shape == (1460, 79))
    assert(y_train.shape == (1460,))
    assert(X_test.shape == (1459, 79))
    
    # test every column in the outputs is of numeric type
    assert(X_train.select_dtypes(exclude = [np.number]).shape[1] == 0)
    assert(X_test.select_dtypes(exclude = [np.number]).shape[1] == 0)