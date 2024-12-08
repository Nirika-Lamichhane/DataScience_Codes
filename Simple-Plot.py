import pandas as pd
import matplotlib.pyplot as plt

file_name='Advertising.csv'

# reading file using pandas
df= pd.read_csv(file_name)

# looking at the data with iloc
df.iloc[7]
# this iloc is integer based location which gives the data of the 8th roww as indexing is from 0

# Creating new dataframe by selecting 7 rows
df_new=df.head(7)

# plotting the graph using the 7 points. If we want to plot all points then just use df only

# using scatter plot
x=df_new['TV']
y=df_new['Sales']

plt.scatter(x,y)

'''
Significance of [[]]  and []
if [[]] is used then the pandas library converts the data though a single column into the 2D
using [] we get the data (i.e. single column) as series or 1D
'''

# Adding axis labels
plt.xlabel ('Tv budget')
plt.ylabel ("Sales")

# Adding plot title
plt.title("The tv vs sales")

# Display plot
plt.show()