
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model as lm
df = pd.read_csv('train.csv')
dftest = pd.read_csv("test.csv")

#%%
sns.heatmap(df.corr(), vmin=0, vmax=1)

# In[2]:

pd.DataFrame(df.corr()["SalePrice"]).sort_values("SalePrice")


# In[78]:
plt.scatter(df["OverallQual"],df["SalePrice"])


# In[4]:

fig = plt.figure(3)
ax = Axes3D(fig)
ax.scatter(list(df["OverallQual"].values),
           list(df["GrLivArea"].values),
           list(df["SalePrice"].values))

plt.show()


# In[6]:

lr = lm.LinearRegression()
X = df[["OverallQual","GrLivArea"]]
y = df.SalePrice
xpred = lr.fit(X, y)

print("R2 = " + str(lr.score(X,y))) #R2 = 0.71


# In[35]:

dftest = pd.read_csv("test.csv")
test = dftest[["OverallQual","GrLivArea"]]
pred=lr.predict(test)

dfpred = pd.concat([dftest.Id,pd.DataFrame(pred,columns=["SalePrice"])],axis=1)
dfpred.loc[pred<30000,"SalePrice"]=30000

dfpred.astype(int).to_csv("submission1.csv", index = False)
print(" With this prediction we got 1419th place with a score of 0.22752")



#%%  2n try: linear regression model with quadratic features

x1 = df.OverallQual**2

X = pd.concat([df.OverallQual**2,
                    df.OverallQual,
                    df.OverallQual * df.GrLivArea,
                    df.GrLivArea,
                    df.GrLivArea**2],    axis = 1)
    
X.columns = ["oq2","oq","oqsp","sp","sp2"]
y = df.SalePrice
test = pd.concat([dftest.OverallQual**2,
                    dftest.OverallQual,
                    dftest.OverallQual * dftest.GrLivArea,
                    dftest.GrLivArea,
                    dftest.GrLivArea**2],    axis = 1)
#%%
model2 = lr.fit(X,y)
print("R2 = " + str(lr.score(X,y)))
pred2 = model2.predict(test) #R2 = 0.77

dfpred2 = pd.concat([dftest.Id, 
                     pd.DataFrame(pred2, columns = ["SalePrice"])],
                    axis = 1)
dfpred2.to_csv("submission2.csv",index = False)
print(" With this prediction we got 1382th place with a score of 0.21050")

#%%














