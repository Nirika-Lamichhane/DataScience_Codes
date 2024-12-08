import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv('poly.csv')
x=df[['x']].values
y=df['y'].values

print(df.head())

# plot x and y for inspecting the graph with actual data
fig,ax=plt.subplots()  # uses default figure and axes size
ax.plot(x,y ,'x')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.set_title('$x$ vs $y$ values')

plt.show()

# fit a linear model on the data
lreg= LinearRegression()
lreg.fit(x,y)

# get the predictions on the entire data 
y_lin_pred=lreg.predict(x)

# here train test split is not done as there isnot use of evaluating the model or if we want to use this model for general datasets

# now polynomial regression
guess_degree=3

# generate a polynomial feature in the entire data
x_poly=PolynomialFeatures(degree=guess_degree).fit_transform(x)

# fit a polynomial model on the data using x_poly as features 
polymodel=LinearRegression(fit_intercept=False)
'''
by default chai hamle linear regression ko garda intercept afai generate garauthem for the correct and good plot of the data
but esma chai .fit_transform le nai hamro x ko data lai a x^2 ko form ma lagdinxa jasle garda ajai arko intercept le redundancy hunxa
so to avoid that fir_intercept false vako

'''
polymodel.fit(x_poly,y)
y_poly_pred=polymodel.predict(x_poly)


# creating the new datasets with no gaps in our prediction line  as well as avoiding the need to create the sorted set of datas

# array of the evenly spaced values
x_l =np.linspace(np.min(x),np.max(x),100).reshape(-1,1)

# prediction on the linespace values
y_lin_pred_l=lreg.predict(x_l)

# polynomial features on the linspace values
x_poly_l=PolynomialFeatures(degree=guess_degree).fit_transform(x_l)

# prediction on the polynomial linspace values
y_poly_pred_l=polymodel.predict(x_poly_l)

# plotting x and y values using plt.scatter
plt.scatter(x,y,s=10,label='Data')

# plot linear regression fit curve 
plt.plot(x_l,y_lin_pred_l,label="Linear fit")

# Also plot the polynomial regression fit curve (using linspace)
plt.plot(x_l, y_poly_pred_l, label="Polynomial fit")

#Assigning labels to the axes
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.show()

# calculating the residuals values
poly_Residuals=y-y_poly_pred

lin_residuals=y-y_lin_pred

#Use the below helper code to plot residual values
#Plot the histograms of the residuals for the two cases

#Distribution of residuals
fig, ax = plt.subplots(1,2, figsize = (10,4))
bins = np.linspace(-20,20,20)
ax[0].set_xlabel('Residuals')
ax[0].set_ylabel('Frequency')

#Plot the histograms for the polynomial regression
ax[0].hist(poly_Residuals, bins,label = 'Polynomial Regression')

#Plot the histograms for the linear regression
ax[0].hist(lin_residuals, bins, label = 'Linear regression')

ax[0].legend(loc = 'upper left')

# Distribution of predicted values with the residuals
ax[1].scatter(y_poly_pred, poly_Residuals, s=10)
ax[1].scatter(y_lin_pred, lin_residuals, s = 10 )
ax[1].set_xlim(-75,75)
ax[1].set_xlabel('Predicted values')
ax[1].set_ylabel('Residuals')

fig.suptitle('Residual Analysis (Linear vs Polynomial)')
plt.show()