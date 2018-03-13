# HousePrices

## Goal
Predict the sales price for each house, based on a set of features.

## Dataset
The [Ames Housing](https://ww2.amstat.org/publications/jse/v19n3/decock.pdf) is the data set used in this project, which was compiled by Dean De Cock for data science education purposes.

## What we did
Since this was one of the first kaggles we participated on we tried a little bit of everything. For this reason, this repository may have become a little messy so here's a description of the major achievements in each file:
* /Feature Engineer.py 
  - Dummy variables
  - Forward Feature Selection using the R2
  - Regression models: Ridge and Lasso 
* /housePrices_onehotencoder.py
  - Applying Onehot Encoding to the categorical features
  - Building linear regression models with them.
* /houseprices.py 
  - Main data analysis. 
  - Plotting the correlation matrix.
  - Feature selection by the correlation to the output and
  - Using the top 3 to make a non-linear regression model.
* /PCA.py
  - tried to apply PCA but the results were not ideal (will try again more thoroughly in the future)
* /src/backward_elimination.py 
  - Feature selection through Backward Elimination method using the p-value
* /src/feat_eng_and_SVR.py 
  - Support Vector Regressor
* /src/OHE_and_ElasticNet.py 
  - Elastic Net regression model
* /src/main
  - main points in a concise and clear code

## Note:
the code files in /src use the kaggle framework built to summarize the pre-processing part of the code in just a couple of lines, ideally in any Kaggle challenge. To use it for the first time you need to follow these quick steps:
  1. Clone the framework repository: https://github.com/AlexGomesDS/Kaggle_project_framework
  2. Add the folder /src in it to sys.path, for example, if you cloned to "C:\GIT\Kaggle_project_framework" run:
```Python
import sys
sys.path.append(r'C:\GIT\Kaggle_project_framework')
```
