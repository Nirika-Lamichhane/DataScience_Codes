import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from itertools import combinations
from prettytable import PrettyTable

df=pd.read_csv('Advertising.csv')

# print(df)

print(df.head()) # default is 5 rows

# creating list for mse values
mse_l=[]

# creating list of all unique combinations of predictors using list comprehension

predictors=['TV','Newspaper','Radio']

cols= []


for r in range(1, len(predictors)+1):

    # combination generation
    comb=combinations(predictors, r)

    for c in comb:
        cols.append(list(c))

print(cols)

for i in cols:

    x=df[i]   # cols afaima list vako vayera eslai [[]] yo garda list  of lists hunxa ra error auxa esma 
    
    y=df['Sales']

    x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=20,random_state=42)

    lreg=LinearRegression()
    lreg.fit(x_train, y_train)

    y_pred=lreg.predict(x_test)

    mse=mean_squared_error(y_test,y_pred)

    mse_l.append(mse)

# using pretty table for displying
t=PrettyTable(['Predictors','MSE' ])

for i in range(len(mse_l)):
    rows=[[cols[i],round(mse_l[i],3)]]
    t.add_rows(rows)
print(t)    

'''
add_rows le chai multiple rows lai ekkaipatak add garna help garxa jun chai list hunu parxa so row lai list garako
edi direct add_row garne ho indivisually vaneni milxa as:
for i in range(len(mse_l)):
    t.add_row([cols[i], round(mse_l[i], 3)])
    esma chai euta euta gardai add hunxa rows haru ani tyo list mai xa so list pheri nabanako

'''