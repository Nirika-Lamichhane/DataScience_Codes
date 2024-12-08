import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.linear_model import LinearRegression

file='Advertising.csv'
df= pd.read_csv(file)

# empty data frame to store the r2 value of each fitted model
df_results=pd.DataFrame(columns=['Predictor', 'R2 Train','R2 Test'])

y=df['Sales']

# for predicting make a function where predictor  is passed as argument
def fit_and_plot_linear(x):
    
    # split the data
    x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.8, random_state=0)

    # training linear model
    lreg=LinearRegression()

    lreg.fit(x_train,y_train)


    # prediction on both train and test data
    y_train_p= lreg.predict(x_train)
    y_test_pred=lreg.predict(x_test)

    # computing R2 values
    r2_train =r2_score(y_train,y_train_p)

    r2_test=r2_score(y_test,y_test_pred)

    # code to plot the data
    plt.scatter(x_train,y_train, color='k',label='Train Data')
    plt.scatter (x_test, y_test, color="r", label="Test data")
    plt.plot (x_train, y_train_p, label="train prediction", color="darkblue",linewidth=2)
    plt.plot(x_test, y_test_pred, label="Test Prediction", color='b', alpha=0.8, linewidth=2, linestyle='--')
    name= x.columns.to_list()[0]
    plt.title(f"Plot to indicate the linear model predictions")
    plt.xlabel(f"{name}",fontsize=14)
    plt.ylabel("Sales",fontsize=14)
    plt.legend()
    plt.show()
    return r2_test, r2_train

# this return statement has no effect on the plot rather it is returned for further observationes i.e. calculating and comparing its value with other model
'''
r2 is the measurement of the variance proportion (i.e. it means the ratio of explained variance to the total variance) 
Y ma variance auxa ani tyo variability lai kasari well explain garxa model le based on X is proportion of variance in model

in math, variance measured how spread out the values of a datasets are from the mean
explained variance vaneko chai model le Y ko variance lai kati account garna sakxa tae ho

if r2=0.9 then 90 % of the variability in Y is explained by X
only 10 % remains unexplained i.e. error

so high r2 then more appreciable model

'''


predictors=['TV','Radio','Newspaper']

for i in predictors:
    r2_test, r2_train = fit_and_plot_linear(df[[i]])
    # append the results to the dataframe
    df_results=df_results._append({'Predictor':i,'R2 Train':r2_train, 'R2 Test':r2_test}, ignore_index=True)

print(df_results)

'''
    while appending the data we used {}dict cause it has keys and values so enusre the error free and proper alinging in the columns
    and list [] ot tuple () both can be used but should be more careful as it requires careful matching of the data order
    set{} it isnot recommended as it is unordered
    '''



# now lets do for the multilinear model using function

def fit_and_plot_multi():

    # getting the predictors and response inside the function so no argument
    x=df[['TV','Radio','Newspaper']]

    x_train, x_test, y_train,y_test=train_test_split(x,y,train_size=0.8, random_state=0)

    lreg=LinearRegression()
    lreg.fit(x_train, y_train)

    y_train_pred=lreg.predict(x_train)

    y_test_pred=lreg.predict(x_test)

    r2_train=r2_score(y_train,y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return r2_train, r2_test

# creates an empty dataframe
df_results=pd.DataFrame(columns=['Model','R2_train','R2_test'])

predictors=['TV','Newspaper','Radio']
for predictor in predictors:
    x=df[[predictor]]
    r2_train, r2_test=fit_and_plot_linear(x)

    # creating a temporary dataframe
    temp_df=pd.DataFrame({
        'Model':[f'linear_{predictor}'],
        'R2_train':[r2_train],
        'R2_test': r2_test
    })

    #concatenate this dataframe to the actual one
    df_results=pd.concat([df_results,temp_df],ignore_index=True)

# calling multi function for all predictors
r2_train_multi, r2_test_multi=fit_and_plot_multi()

# creating temporary data frame for multi linear model
temp_df=pd.DataFrame({
    'Model':'linear multi',
    'R2_train': [r2_train_multi],
    'R2_test': [r2_test_multi]

})

# concatenate
df_results=pd.concat([df_results, temp_df],ignore_index=True)

print(df_results)