
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib qt5')
from mpl_toolkits.mplot3d import Axes3D
os.listdir('.')


# In[2]:

df = pd.read_csv('train.csv')
corrmat=df.corr()
corrmat


# In[65]:

df2=pd.DataFrame(corrmat["SalePrice"])
df2.sort_values("SalePrice")


# In[67]:

asd = sns.heatmap(df.corr(), vmin=0, vmax=1)


# In[3]:

plt.scatter(df["OverallQual"],df["SalePrice"])


# In[78]:

#plt.scatter(df["OverallQual"],df["GrLiveArea"],df["SalePrice"])


# In[4]:

fig = plt.figure(3)
ax = Axes3D(fig)
ax.scatter(list(df["OverallQual"].values),
           list(df["GrLivArea"].values),
           list(df["SalePrice"].values))

plt.show()


# In[6]:

from sklearn import linear_model as lm


# In[16]:

lr = lm.LinearRegression()
X = df[["OverallQual","GrLivArea"]]
y = df.SalePrice
xpred = lr.fit(X, y)


# In[25]:

print lr.score(X,y)


# In[35]:

dftest = pd.read_csv("test.csv")
test = test[["OverallQual","GrLivArea"]]


# In[39]:

pred=lr.predict(test)


# In[76]:

pred<0


# In[116]:

dfpred = pd.DataFrame([dftest.Id, pred],
                      dtype=object)
dfpred = dfpred.transpose()
dfpred.columns=["Id", 'SalePrice']
dfpred.loc[pred<30000,"SalePrice"]=30000


# In[117]:

dfpred.astype(int).to_csv("submission1.csv", index = False)


# In[ ]:




# In[119]:

df.DataFramdf.OverallQual**2

