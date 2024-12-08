import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures


df=pd.read_csv('poly.csv')
print(df.head())

# getting the predictor and response values
x=df[['x']].values
y=df['y'].values

# plot x and y to inspect data
fig, axes=plt.subplots(figure=(10,8))

axes.plot(x,y ,'x')
axes.set_xlabel('$x$ values')
axes.grid(True)
axes.set_ylabel('$y$ values')
axes.set_title('$y$ vs $x$')
plt.show()

# split the train test model
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

lreg=LinearRegression()
lreg.fit(x_train,y_train)
y_pred=lreg.predict(x_test)

# plot of the model fitted by linear regression
figs,axes=plt.subplots()
axes.plot(x,y,'x',label='data')
axes.set_xlabel('$x$ values')
axes.set_ylabel('$y$ values')

axes.plot(x_test, y_pred,'--',label='linear model predictions')
plt.legend()
plt.show()

guess_degree=3

def get_poly_pred(x_train,x_test,y_train,degree=1): # y_test isnot used here because we are only generating predictions here and we arenot calculating the accuracy or evaluating them
    # generate polynomial features on the train data
    x_poly_train=PolynomialFeatures(degree=degree).fit_transform(x_train)


    # generate polynomial fetaures on the test data
    print(x_train.shape, x_test.shape,y_train.shape)
    x_poly_test=PolynomialFeatures(degree=degree).fit_transform(x_test)

    # initialize the model to perform polynomial regression
    polymodel=LinearRegression()

    # fit the model on the polynomail transformed train data
    polymodel.fit(x_poly_train, y_train)

    # predict on the entire polynomial transformed test data
    y_poly_pred =polymodel.predict(x_poly_test)
    return y_poly_pred

y_poly=get_poly_pred(x_train,x_test,y_train,degree=guess_degree)
print(y_poly)

# helper code to visulaize the results
idx=np.argsort(x_test[:,0])  
'''
esle euta column lai sort garxa kinaki yesma euta matra predictor xa with number of samples
esle chai hamro data lai ascending order ma sort garna ko lagi index provide garxa numbers haru ko from smaller to higher
ani tyo idx ma save hunxa
suppose data is x=[[1], [3],[2]] in colmun format then teslai x_test[:,0] le row list ko form ma lerauxa
ani index chai [0,2,1] hunxa for sorting
'''
# use the index above for the appropriate values of y
x_test=x_test[idx]
y_test=y_test[idx]

# linear predicted values
y_pred=y_pred[idx]

# non linear predicted values
y_poly=y_poly[idx]

# hamle predicted datas haru lai chai test wala use garera sorting garem for the smooth curve

# plotting x and y values using plt.scatter

plt.scatter(x,y,s=10, label='Test Data') # this plots the actual datas and target values using sctter dots of size 10 labeled as test data

# plot the linear regression fit curve
plt.plot(x_test,y_pred,label='Linear fir',color='k')

# plot the polynomial regression fit curve
plt.plot(x_test,y_poly,label='Polynomial fit',color='red',alpha=0.7)

# plot is used as it plot the st continuous line whereas the scatter is used to plot the individual data points

# assigning labels to the axes
plt.xlabel('X values')
plt.ylabel('Y valyes')
plt.legend()
plt.show()

# calculate the residual values for the polynomial model
poly_residuals=y_test-y_poly

# caluclate residuals for the linear model
lin_residuals=y_test-y_pred

# helper code to plot the residual values
# plot histograms of the residuals for two cases

fig,ax = plt.subplots(1,2,figsize=(10,4))
bins=np.linspace(-20,20,20)

ax[0].set_xlabel('Residuals')
ax[0].set_ylabel('Frequency')

# plot the histograms for the polynomial regression
ax[0].hist(poly_residuals,bins,label='poly residuals', color='r',alpha=0.9)

# plot the histograms for the linear regression
ax[0].hist(lin_residuals, bins, label='linear residuals', color='k', alpha =0.9)

ax[0].legend(loc='upper left')

# distribution of predicted values with the residuals
ax[1].hlines(0,-75,75, color='k', ls='--', alpha =0.3, label="zero residuals")
ax[1].scatter(y_poly, poly_residuals, s=10, color='b', label='Polynomial predictions')
ax[1].scatter(y_pred, lin_residuals, s = 10, color='#EFAEA4', label='Linear predictions' )
ax[1].set_xlim(-75,75)
ax[1].set_xlabel('Predicted values')
ax[1].set_ylabel('Residuals')
ax[1].legend(loc = 'upper left')
fig.suptitle('Residual Analysis (Linear vs Polynomial)')
plt.show()