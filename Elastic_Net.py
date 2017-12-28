# -*- coding: utf-8 -*-
"""
This code tries the elastic-net algorithm with all the possible variables.
Which means both numerical and categorical with label encoding, and one-hot encoding
"""

#%% 1.1 Import libraries
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
#%matplotlib qt5
from mpl_toolkits.mplot3d import Axes3D
#from sklearn import linear_model as lm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer
from sklearn.metrics import mean_squared_error

#%% 1.2 import data
train_dataframe = pd.read_csv('train.csv')
test_dataframe = pd.read_csv("test.csv")

#%% Preprocess the data sets
class preprocessing:
    def prepare(self, training_data_frame):
        # Deal with missing values by using the mean
        self.imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        self.imputer.fit(training_data_frame.select_dtypes(include=[np.number]).values)
        self.cat_le = {}
        for col in training_data_frame.select_dtypes(exclude=[np.number]).columns:
            self.cat_le[col] = LabelEncoder().fit(training_data_frame[col].fillna('NaN'))
    
    def prepare_and_cook(self, training_data_frame):
        self.prepare(training_data_frame)
        return self.cook(training_data_frame)
    
    def cook(self, df):
        #initialize engineered dataframe (must use a copy otherwise we would affect the original DF)
        df_eng = df.copy()
        
        # separate into categorical and numerical features
        df_eng_num = df_eng.select_dtypes(include=[np.number])
        df_eng_cat = df_eng.select_dtypes(exclude=[np.number]).fillna('NaN')
        
        # label encode categorical features
        for col in df_eng_cat.columns:
            column_transf = self.cat_le[col]
            df_eng_cat[col] = column_transf.transform(df_eng_cat[col])
                    
        #Replace nan's with mean
        cols = df_eng_num.columns
        idx = df_eng_num.index
        df_eng_num = pd.DataFrame(self.imputer.transform(df_eng_num))
        df_eng_num.columns = cols
        df_eng_num.index = idx
        
        if "SalePrice" in df.columns:
            # Following commands only apply to train set
            df_eng_num["logSalePrice"] = np.log(df.SalePrice)
        
        return df_eng.append(df_eng_cat)


#%%
data_preprocessor = preprocessing()
eng_train_dataset = data_preprocessor.prepare_and_cook(train_dataframe)
eng_test_dataset = data_preprocessor.cook(test_dataframe)
