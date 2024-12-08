import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

file="Advertising.csv"

# creating data frame 
df = pd.read_csv(file)

# print the dataset and take a look in it
df.head() # sets 5 rows automatically if nothing is mentioned

'''
We have many predictors but using TV column as a predictors and getting the data from rows 5 to 13
using sales data as the response
'''
x=df.TV.iloc[5:13]
y=df.Sales.iloc[5:13]

# using the numpy attributes we can sort the data to get indices ordered from lowest to highest TV values
idx =np.argsort(x)

# Get the predictor and the response data in the order by the give idx values
x_sorted = x.iloc[idx].values
y_sorted =y.iloc[idx].values

'''
Define the function that finds the index of the nearest neighbor and returns the value of the nearest neighbor
note that this is just for k=1 where distance is simply the absolute value

'''
def find_nearest(array,value):
    # here array is the sequence of numerical data and values are the target number to which we want to find the closet value in array
    idx=pd.Series(np.abs(array-value)).idxmin()

    # returing the nearest neighbor index and valye
    return idx, array[idx]

# Creating some synthetic x-values (might not be in actual dataset)
x_syn=np.linspace(np.min(x),np.max(x))

# Initialize the y-values as 0 for all the values equal to the length of x i.e. arrays of 0
y_syn=np.zeros(len(x_syn))

'''
Synthetic x banauda kheri hami sanga discrete data jastai x=[1,2,3] huda plot discountinous hunxa
ani ramrari visualization ra interpolate garna sakidaina so synthetic garda smooth plot auxa
i.e. plot original data samma matra simit hudaina ra esle continous rannge banaidinxa jasle garda hamle between the data joints ko pani 
plot thapauna sakxam
yo nagarda plot ekdamai dicontinous hunxa ra original data ma matra simit hunxa

here y_syn acts as placeholder jasma chai hamle paxi x_Syn ko corresponding y vales halna sakxam for plotting
and further analysis.
'''
# Applying the knn algorithm to predict y value for the x value
for i , xi in enumerate(x_syn):
    '''
    hamle esma enumerate use garxum kinaki hamlai normal for loop le values ma matra iterate garxa of a array.
    if we need index of each values, enumerate provides both index and the values
    Without enumerate(), you would need to manually track the index in a separate variable, making the code more complex.
    '''

    # Get the sales values closest to the given x values
    y_syn[i]=y[find_nearest(x,xi)[0]]

    '''
    this works as: x_syn ko value chai as a xi i.e. values pass hunxa find_nearest function ma ani x chai array ko rup ma janxa
    array ma esle xi ko nearest value patta lagauxa ra suruko index ra value lai as a tuple retur garxa
    tesaile (2,5) yo auda 2 index ho vane 5 chai nearest value of x ma xi ko 
    so y patta lagauna 2 index nikalna [0] rakheko as 2 index is in 0 index of the tuple
    ani teslai y_syn ma rakheko in each iteration

    '''
# Plotting the data

plt.plot (x_syn, y_syn, '-.')

# Plotting original data using black x's
plt.plot(x ,y ,"kx")

plt.title("TV vs Sales")
plt.xlabel("TV budget")
plt.ylabel("Sales")

plt.show()