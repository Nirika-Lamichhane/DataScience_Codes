# multicollinearity vaneko chai independent variables haru ekarka sanga dherai nai correlated vaye vane hunxa ani esle chai value prediction ma problem hunxa
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pprint import pprint
import seaborn as sns

df= pd.read_csv('colinearity.csv')
print(df.head())

# taking the x and y values from the dataset
X=df.drop(['y'],axis=1)

y=df['y'].values.ravel()

# model fit garda y chai 1D hunu parxa so teslai convert garna 1D ma ravel() use gareko
print("shape of x: ", X.shape)
print("shape of y: ", y.shape)
'''
this gives the shape in the foem as (rows,columns)
where row is the number of sample data and columns represents the predictors
so in x it is 2D as there are number of predictors but in y it is 1D as there is only one column

'''

# creating coef for the simple regression
linear_coef=[]  # list for storing all the individual coefficients of the predictor

for i in X.columns:
    x=X[[i]]

    lreg=LinearRegression()
    lreg.fit(x,y)

    linear_coef.append(lreg.coef_)

print(linear_coef)

# multilinear regression using all the variables

mul=LinearRegression()
mul.fit(X,y)
multi_coef=mul.coef_

print(multi_coef)  # by doing this it prints the array

# for systematic printing of the coefficient and Beta value we do the following:

print('Beta values of different predictors by simple linear regression i.e. one predictor at a time:: ')
for i in range(len(linear_coef)):
    # esma range(4) direct lekhna sakinxa if we know that there are only 4 columns in it otherwise this is general method

    pprint(f'Value of Beta{i+1} = {linear_coef[i][0]:.4f}')  # [0] le hamro array ko form ma vayeko data lai scalar ko rup ma transform gardinxa

print('Beta values by multi-linear regression on all variables:: ')
for i in range(4):
    pprint(f'Value of Beta{i+1}={round(multi_coef[i],2)}')  # round the coefficient to 2

# code for visulaizing the heatmap of covariance matrix
corrMatrix =df[['x1','x2','x3','x4']].corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
     
'''
collinearity is measured by the correlation matrix which measures the strength and direction of linear relationship between two variables
it mesaures how the one variable is related to ohter and ranges from -1 to +1 and is calculated by pearson correlation coefficient

'''