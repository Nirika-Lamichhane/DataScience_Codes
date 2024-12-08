# here we will find the affect of different scales in our model and how it affects the model performance

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

df=pd.read_csv('Advertising.csv')
df.head()

# fiting multiple linear regression using all predictors
x=df.drop('Sales',axis=1)
y=df.Sales.values

lm=LinearRegression()
lm.fit(x,y)

# report coefficients and the r2 value
print("model coefficients")

# loop through column and coefficient to print them
for col, coef in zip(x.columns,lm.coef_):
    print(col,":",coef)

# print r2 score
print("score: ",lm.score(x,y))
'''
esma x.columns le predictors ko name haru lai janauxa and lm.coef le chai array of coefficients
from linear regression model janauxa 
harek linear model ma harek feature ko corresponding coef value hunxa jun chai array ko roop ma auxa
zip function le chai each column name lai corresponding coef sanga pair garera tupke for form ma dinxa
jasko first item column name ra second item chai coefficient ho

'''
# scale up the dataframes by 1000 times and refit and see change in the r2 score of model

df*=1000
df.head()
x=df.drop('Sales',axis=1)
y=df.Sales.values
lm=LinearRegression()
lm.fit(x,y)

print("Model coefficients:: ")
for col, coef in zip(x.columns, lm.coef_):
     print(f'{col:>9}: {coef:>6.3f}')
     
print(f'\nR^2: {lm.score(x,y):.4}')

'''
the coefficient and the loss are the same as the linear regression coefficient are invariant under scaling
because we have scaled both predictors and response

'''
# plot using horizontal bar plot
plt.figure(figsize=(8,3))

# column name to be displayed on the y axis
col=x.columns

# coefficient values from our fitted model
coefs=lm.coef_

# create the horizontal barplot
plt.barh(col,coefs)

# dotted semitransparent black vertical line at 0
plt.axvline(0, c='k', ls='--',alpha=0.5)

# always label the axes
plt.xlabel("Coefficient Values")
plt.ylabel("Predictor")


# creating the title
plt.title("Coefficient of linear model predicting Sales \n from Newspaper, Radio and Tv Advertising Budgets (in Dollars)")
plt.show()

# creating a new dataframe by changing the scales

x2=pd.DataFrame()
x2['TV(Rupee)']=200*df['TV'] # converted to srilankan rupees
x2['Radio(Won)']=1175*df['Radio']
x2['Newspaper(Cedi)']=6*df['Newspaper']

lm2=LinearRegression()
lm2.fit(x2,y)

# reporting coefficients and mse
print(f'{"Model Coefficients:>16"}')

'''
yo f string ho esma chai characters haru i.e. string literals lai {} bhitra rakhnu parxa
>16 esle chai hamro sting lai right alignment garera 16 spaces provide gardinxa
tyo 16 spaces vanda greater hamro text width vayo vane jasta ko testai print hunxa
otherwiseit will be padded with the spaces on the left.
space xodxa left ma jati free xa teti lai

'''
for col, coef in zip(x2.columns, lm2.coef_):
    print(f'{col:>16}:{coef:>8.5f}')

print(f'\n R^2: {lm2.score(x2,y):.4}')

'''
here in formatting in coef: >8 it is used for right alignment of coef with 8 space and .5f is used to denote that
the number of decimal numbers will be 4 . denotes decimal point 4 denotes no of decimal points and f denotes the floating number
while in R2 calculation only .4 is used because it helps to print the number of significant figures after the decimal point but 
not the decimal numbers
if .4f is used then 0.000101= 0.0001
if not then 0.0001010 = 0.0001010

'''
# now lets plot the coefficient and predictor to see how the scaling affected the coefficient

plt.figure(figsize=(8,3))
plt.barh(x2.columns, lm2.coef_)
plt.axvline(0, c='k', ls='--', alpha =0.9)
plt.ylabel('Predictor')
plt.xlabel("Coefficient values")
plt.title('Coefficients of Linear Model Predicting Sales\n from Newspaper, '\
            'Radio, and TV Advertising Budgets (Different Currencies)')

plt.show()

# we cannot determine and compare th emse values of the regression model having different scales and currencies

# plotting the two plots in the shared x axis to see the difference in actual and the scaled data
fig ,axes = plt.subplots(2,1, figsize=(8,6), sharex=True)

axes[0].barh(x.columns, lm.coef_)
axes[0].set_title("Dollars")
axes[1].barh(x2.columns, lm2.coef_)
axes[1].set_title("Different Currencies")

for ax in axes:
    ax.axvline(0,c='r',ls='--', alpha=0.8)

axes[0].set_ylabel('Predictor')
axes[1].set_xlabel('Coefficient values')

plt.show()