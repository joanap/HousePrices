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
from sklearn.metrics import mean_squared_error

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

#%% Function that receives vector of features to use to train the linear regr. model and returns R2
lr = lm.LinearRegression()

def calc_r2(vfeats):
    x = dfclean[vfeats]
    model = lr.fit(x,y)
    r2 = model.score(x,y)
    n = len(x)
    k = len(x.columns)
    return 1 - (1-r2)*(n-1)/(n-k-1)

#%%

vfeats = []
res= []
m = len(dfclean.columns)
for num_feats in range(m):
    l_r2 = []
    for i in range(m):
        if not i in vfeats:
            l_r2.append([i,calc_r2(vfeats + [i])])
    nplr2 = np.array(l_r2)
    R2_max = nplr2[:,1].max()
    arg_R2_max = nplr2[np.argmax(nplr2[:,1]),0].astype(int)
    print("{} features. Max adj R2 of {}. New feat added was {}:{}".format(num_feats + 1, R2_max,arg_R2_max,numdf.columns[arg_R2_max]))
    vfeats.append(arg_R2_max)
    res.append([num_feats,R2_max])

res = np.array(res)
plt.scatter(res[:,0],res[:,1])
X=vfeats[:19]

#%% predict using 19 first features
FinalModel = lr.fit(dfclean[X],y)

pred=np.exp(lr.predict(df_testclean[vfeats[:14]]))

dfpred = pd.concat([dftest.Id,pd.DataFrame(pred,columns=["SalePrice"])],axis=1)
dfpred.to_csv("submission_14feats.csv", index = False)
### "Your submission scored 0.14355, "  1083rd place!!!!

#%% predict using ridge regression
reg = lm.RidgeCV(alphas = [0.01,0.02,0.03,0.5,0.7,.9],normalize=True)
reg.fit(dfclean[X],y)
reg.alpha_

pred=np.exp(reg.predict(df_testclean[vfeats[:19]]))

dfpred = pd.concat([dftest.Id,pd.DataFrame(pred,columns=["SalePrice"])],axis=1)
dfpred.to_csv("submission_ridge.csv", index = False)
# score=0.14298 even worse booo
#%% Let's try Lasso
regl = lm.LassoCV(alphas = [1e-5,1e-4,1e-3,1e-2,1e-1],normalize=True)
regl.fit(dfclean[X],y)
regl.alpha_

pred=np.exp(regl.predict(df_testclean[vfeats[:19]]))

dfpred = pd.concat([dftest.Id,pd.DataFrame(pred,columns=["SalePrice"])],axis=1)
dfpred.to_csv("submission_Lasso.csv", index = False)

#cross validation
def crossValidation(inputSet):
    kf = KFold(n_splits=10)
    for train, test in kf.split(inputSet):
        #print("%s %s" % (train, test))
        lr.fit(inputSet.loc[train], y.loc[train])
        pred_cv = lr.predict(inputSet.loc[test])
        mean_squared_error(y.loc[test], pred_cv)

crossValidation(dfclean[vfeats[:14]])
