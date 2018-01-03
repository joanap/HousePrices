# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 23:33:01 2018

@author: Alexandre

This class implements the following preprocessing methods:
    - Replace NaN values by their mean / most common label
    - Encode the categorical features according to the mean of the output for each category
    
"""


#%% Import onyl the necessary libraries
from features.preprocessor.abstract_preprocessor import AbstractPreprocessor 

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#%% Implementing class with our one version of the preprocessing methods

class Mean_Mode_Preprocessor(AbstractPreprocessor):
    
    # replace missing numerical values with the mean
    def _calc_missing_num_replacements(self, col):
        return col.mean()

    # replace missing labels with the most common one
    def _calc_missing_cat_replacements(self, col):
        return col.value_counts().index[0]    
    
    def _gen_cat_col_encoder(self, col):
        new_df = pd.DataFrame()
        new_df['vals'] = col.unique()
        new_df.index = new_df.vals
        new_df
        return LabelEncoder().fit(col.fillna('NaN').append(pd.DataFrame(['NaN'])))
    
    # Check if this is training set by looking for the output column called "SalePrice"
    def _is_trainning_set(self, dataframe):
        return "SalePrice" in dataframe.columns
    
    # Don't create any new features for both datasets
    def _feat_eng(self, dataframe):
        pass
    
    # Create a column with the log of the saleprice
    def _feat_eng_train(self, dataframe):
        dataframe["logSalePrice"] = np.log(dataframe.SalePrice)



#%% Function to assign a value to a category equal to the average sale price in that category
def attribute_creater(feature_name, df):
    new_df = pd.DataFrame()
    new_df['vals'] = df[feature_name].unique()
    new_df.index = new_df.vals
    new_df['means'] = df[[feature_name, 'SalePrice']].groupby(feature_name).mean()['SalePrice']
    new_df = new_df.sort_values('means')
    new_df = new_df['means'].to_dict()
    
    for cat, o in new_df.items():
        df.loc[df[feature_name] == cat, feature_name+'_E'] = o
        
    return df   

#%% Testing

if __name__ == '__main__':
    train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='Id')
    test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='Id')

    data_preprocessor = Mean_Mode_Preprocessor()
    eng_train_dataset = data_preprocessor.prepare(train_dataframe)
    eng_test_dataset = data_preprocessor.cook(test_dataframe)
