# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 01:33:02 2017
@author: ASSG
"""
# In[1]:
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
#%matplotlib qt5
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model as lm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer

df = pd.read_csv('train.csv')
dftest = pd.read_csv("test.csv")

#Preproces the data
# Use the log of saleprice!!
df["logSalePrice"] = np.log(df.SalePrice)
#select only relevant feats. (numeric except the outputs)
numdf = df.select_dtypes(include=[np.number]).drop(["Id","logSalePrice","SalePrice"],1)
numdf_test = dftest.select_dtypes(include=[np.number]).drop(["Id"],1)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#train the imputer (in this case get the mean):
imputer = imputer.fit(numdf.values)
#apply the imputer transformation, i.e. replace NaN by the mean, don't forget to apply exactly the same transf to the test set!!!
x = imputer.transform(numdf.values)
xt = imputer.transform(numdf_test.values)

dfclean = pd.DataFrame(x)
df_testclean = pd.DataFrame(xt)

y = df["logSalePrice"]


#%% imputing missing values
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
    
catdf = df.select_dtypes(include=[object])
catdf_test = dftest.select_dtypes(include=[object])

X = pd.DataFrame(catdf)
xt = DataFrameImputer().fit_transform(X)

X_test = pd.DataFrame(catdf_test)
xt_test = DataFrameImputer().fit_transform(X_test)

# adding dummies

dfbinary = pd.get_dummies(xt)
dfbinary_test = pd.get_dummies(xt_test)

dfclean = pd.concat([dfclean, dfbinary], axis=1)
df_testclean = pd.concat([df_testclean, dfbinary_test], axis=1)



#%% Function that receives vector of features to use to train the linear regr. model and returns R2
lr = lm.LinearRegression()

def calc_r2(vfeats):
    column_name = dfclean.columns[vfeats]
    x = dfclean[column_name]
    model = lr.fit(x,y)
    r2 = model.score(x,y)
    n = len(x)
    k = len(x.columns)
    return 1 - (1-r2)*(n-1)/(n-k-1)

#%% 

vfeats = []
res= []
m = len(dfclean.columns)
previous_r2 = 0
for num_feats in range(m):
    l_r2 = []
    for i in range(m):
        if not i in vfeats:
          l_r2.append([i,calc_r2(vfeats + [i])])
    nplr2 = np.array(l_r2)
    R2_max = nplr2[:,1].max()
    arg_R2_max = nplr2[np.argmax(nplr2[:,1]),0].astype(int)
    if R2_max < previous_r2:
        break
    else:
        previous_r2 = R2_max
        
    # print("{} features. Max adj R2 of {}. New feat added was {}:{}".format(num_feats + 1, R2_max,arg_R2_max,numdf.columns[arg_R2_max]))
    print("{} features. Max adj R2 of {}. New feat added was {}:{}".format(num_feats + 1, R2_max,arg_R2_max,dfclean.columns[arg_R2_max]))
     
    vfeats.append(arg_R2_max)
    res.append([num_feats,R2_max])
res = np.array(res)
plt.scatter(res[:,0],res[:,1])

#%% predict using best features

X=vfeats[:129]
FinalModel = lr.fit(dfclean[X],y)

pred=np.exp(lr.predict(df_testclean[vfeats[:129]]))

dfpred = pd.concat([dftest.Id,pd.DataFrame(pred,columns=["SalePrice"])],axis=1)
dfpred.to_csv("submission_best_feats.csv", index = False)
