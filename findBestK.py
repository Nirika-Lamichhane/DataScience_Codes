import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

file="Advertising.csv"

df=pd.read_csv(file)

x=df[["TV"]].values
y=df["Sales"].values

x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=80, random_state=42)

k_min=1
k_max=70

# Creating a list of integer k values
k_list=np.linspace(k_min,k_max,num=70,dtype=int)

fig,ax=plt.subplots(figsize=(10,6))

knn={} # this is dict
j=0 # variable for altering the linewidth of values knn models

for i in k_list:

    # create knn regression model for the current k
    model=KNeighborsRegressor(n_neighbors=int(i))
    # n_neighbors is the attribute of the model

    # fit the model
    model.fit(x_train, y_train)

    # predict the data
    y_pred=model.predict(x_test)

    # mse of test data
    MSE =mean_squared_error(y_test,y_pred)

    knn[i]=MSE

    # pltting the data and various knn model predictions
    colors=['grey','r','b']
    if i in [1,10,70]:
        xvals = np.linspace(x.min(), x.max(),100).reshape(-1,1)
        ypreds=model.predict(xvals)
        ax.plot(xvals, ypreds,'-',label = f'k = {int(i)}',linewidth=j+2,color = colors[j])
        j+=1
        
ax.legend(loc='lower right',fontsize=20)
ax.plot(x_train, y_train,'x',label='test',color='k')
ax.set_xlabel('TV budget in $1000',fontsize=20)
ax.set_ylabel('Sales in $1000',fontsize=20)
plt.tight_layout()

plt.show()

# for the relation between k values and mse
plt.figure(figsize=(8,6))

k_values=list(knn.keys())
mse_values=list(knn.values())

plt.plot(k_values, mse_values,'k.-',alpha=0.5, linewidth=2)

# alpha denotes the transparency of the elements
# Set the title and axis labels
plt.xlabel('k',fontsize=20)
plt.ylabel('MSE',fontsize = 20)
plt.title('Test $MSE$ values for different k values - KNN regression',fontsize=20)
plt.tight_layout()

plt.show()