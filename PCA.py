# tried to apply PCA, didn't work...
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(3)
ax = Axes3D(fig)
ax.scatter(list(dfclean["OverallQual"].values),list(dfclean["GrLivArea"].values),list(y.values))
plt.show()

scaler = StandardScaler().fit(dfclean)
df_normalized = scaler.transform(dfclean)
df_reducer = PCA(n_components = 2).fit(df_normalized)
df_reduced = df_reducer.transform(df_normalized)

fig = plt.figure(4)
ax = Axes3D(fig)
super_x1 = df_reduced[:,0]
super_x2 = df_reduced[:,1]
ax.scatter(super_x1,super_x2,y)
plt.show()

X = [super_x1**2,
                    super_x1,
                    super_x1 * super_x2,
                    super_x2,
                    super_x2**2]
regl = lm.RidgeCV(alphas = [1e-3,3e-3,1e-2,3e-2,1e-1,3e-1],normalize=True)
regl.fit(np.transpose(X),y)
print(regl.alpha_)


plt.scatter(df_reduced,y)
dftestnorm = scaler.transform(df_testclean)
df_testclean_reduced = df_reducer.transform(dftestnorm)

super_xt1 = df_testclean_reduced[:,0]
super_xt2 = df_testclean_reduced[:,1]
Xt = [super_xt1**2,
                    super_xt1,
                    super_xt1 * super_xt2,
                    super_xt2,
                    super_xt2**2]
pred=np.exp(regl.predict(np.transpose(Xt)))

dfpred = pd.concat([dftest.Id,pd.DataFrame(pred,columns=["SalePrice"])],axis=1)
dfpred.to_csv("Submission_SuperFeatures.csv", index = False)
