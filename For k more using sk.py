import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

file='Advertising.csv'
df=pd.read_csv(file)

x=df[['TV']].values
y=df ['Sales'].values

# Split the dataset into training and the testing data
x_train , x_test , y_train , y_test = train_test_split(x,y,train_size=0.6, random_state=42)

k_min=1 # This means the data sets reponse will be predicted using the single nearest neighbors
k_max=70 # This means it uses 70 nearest points for prediction


'''
 k value vaneko chai number of the neighbors used to predict the reponse values for the given data point
k value dherai huda chai small details of data miss huna sakxa so we must consider cross validation 
matlab ki different k values ma model lai train garne

'''
# Creating the list of interger k values between max and min using linspace
k_list= np.linspace(1,10,70)

'''
linspace le chai 1 bata 10 samma 70 ota equal spacing numbers provide garxa
equal spacing vako vayera 1 10 int vayeta pani 70 ota numbers chai floating hunxa
esle list haina NumPy array provide garxa jaslai hamle afai int ma convert garnu parne hunxa

'''

# Set the grid to plot the values
fig, ax = plt.subplots(figsize=(10,6))  #(width, height)

'''
fig refers the entire figure or canvas which can contain multiple plots or subplots
ax refers the single axes object where the actual data visualization (plot) will be drawn. 
It's like the “plot area” within the figure.
'''

# Variable used to alter the linewidth of each plot
j=0    # it controls how thick the line is in the plot


x_train = x_train.reshape(-1, 1)  # For single feature
x_test = x_test.reshape(-1, 1)

'''
-1 vaneko numpy ko reshape method ho esle chai automatically dimension patta lagauxa based on the total number of elements in array
edi ma sanga [2,3,4,5] xa vane 4 ota elemers so (-1,1) garda (4,1) hunxa i.e. 4 rows ra 1 column 
esle column chai 1 banauxa
ani (1,-1) huda 1 row 6 column hunxa i.e. yesle simply table 2D format ma dine ho datas

The total number of elements in an array is calculated by multiplying the size of all dimensions.
aba esma hamle (2,2) garna milxa but (1,3) esto haru garnu mildaina as 1x3=3 not equal to 4


'''

# Validate k_list to ensure it contains positive integers
k_list = [1, 10, 70]  # Define k_list with integers
for k_value in k_list:
    # Check if k_value is a valid positive integer
    if not isinstance(k_value, int) or k_value <= 0:
        raise ValueError(f"Invalid k_value: {k_value}. It must be a positive integer.")

    # Create and train the kNN Regressor
    model = KNeighborsRegressor(n_neighbors=k_value)
    model.fit(x_train, y_train)  # fit = train

    # Predict
    y_pred = model.predict(x_test)
    
    # Helper code to plot the data along with the model predictions
    colors = ['grey','r','b']
    if k_value in [1,10,70]:
        xvals = np.linspace(x.min(),x.max(),100).reshape(-1,1)
        ypreds = model.predict(xvals)
        ax.plot(xvals, ypreds,'-',label = f'k = {int(k_value)}',linewidth=j+2,color = colors[j])
        j+=1

    

        
ax.legend(loc='lower right',fontsize=20)
ax.plot(x_train, y_train,'x',label='train',color='k')
ax.set_xlabel('TV budget in $1000',fontsize=20)
ax.set_ylabel('Sales in $1000',fontsize=20)
plt.tight_layout()

'''
The function plt.tight_layout() in Matplotlib is used to automatically adjust 
the spacing of plot elements (such as axes, labels, titles, and legends) to 
ensure that everything fits within the figure area without overlap.
'''
plt.show()