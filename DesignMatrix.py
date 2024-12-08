import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# loading the credit data
df= pd.read_csv('credit.csv')
print(df.head())

# selecting predictors using the drop attributes
x=df.drop('Balance',axis=1)
# axis 1 means column and axis 0 is row
y=df['Balance']

'''
List Comprehension in python is used to create the list in a concise way
harek iterable items jastai tuples range ra list ko harek items ma expression ra condition apply garera list banauxa

new_list=[expression for item in iterable if condition]
range(start, stop, step)
it includes the start and increments by step until the value reaches stop but doesnot include stop

for squaring or cubing we can use different notations but in mathematical form we use:
# Squaring
print(x**2)  # Simple and readable
print(x*x)   # Still fine for squaring

# Cubing
print(x**3)  # Clean and scalable
print(x*x*x) # More cumbersome for higher powers

# Square root
print(x**0.5)  # Convenient
# Can't directly represent this with x*x

eg 
squares=[x**2 for x in range(5) if x%2==0]
print(squares)

Nested list 
here The outer list comprehension (for i in range(5)) creates rows, 
and the inner list comprehension (for j in range(5)) adds the elements in each row.
matrix = [[(i + j) for j in range(5)] for i in range(5)]
print(matrix)


'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=LinearRegression()
# checking if all features fit in the model or not
try:
    test_model=model.fit(x_train,y_train)
except Exception as e:
    print('Error!:',e)  
    '''
        Exception is the base class for all built in exceptions in python
        esko matlab ki esle jun errors lai ni catch garxa jasto type ko vayeni
        catch(...)     '''

print(df.dtypes)

# excluding the categroial and including only numeric features
nf= x_train.select_dtypes(include=['number']).columns.tolist()
lreg=LinearRegression()
lreg.fit(x_train[nf],y_train)



'''
numeric features can also be obtained by the list comprehension as:
numeric_features = [col for col in x_train.columns if pd.api.types.is_numeric_dtype(x_train[col])]
print(numeric_features)

here in fit x_train[nf] is used instead of using nf only is because nf is the list of the column but we need the array or the 2D data
so doing x_train[nf] creates the the dataframe of the columns which are numeric as in the nf list


pd.api.types is a module in pandas jasle chai kun data type ho vanera check garna help garxa
esma various functions hunxa eg is_numeric_dtype(), is_string_dtypes(),etc

pandas API is the set of tools and commands of the library that enables data analysis and manipulation

panda series vaneko chai 1D array ho i.e. column ma hunxa ani harek ko index hunxa
eg 
import pandas as pd
data=[10,20,30]
series=pd.Series(data)  yo garda index 0 bata 1 2 3 hudai afai janxa
series=pd.Series(data, index['a','b','c','d']) yo garda index 0 bata huna ko satto a b c hudai janxa

print(series)
'''

# r2 score

train_score =lreg.score(x_train[nf],y_train)
test_score=lreg.score(x_test[nf],y_test)

print("Train R2: ",train_score)
print("Test Score: ",test_score)

'''
here .score() is used instead of r2_score() because r2_score use garna lai y ko value exclusively predict garnu parxa
but score le afai y predict garera value nikaldinxa so it is more preferred

'''

# another way of getting the r2_score

y_train_pred =lreg.predict(x_train[nf])
'''
esma nf use nagarda value error auxa as train garda ra model fit  garda numeric column ma matra garya xa ani esma sabai garna khojda
overfitting hunxa
'''
r2_train =r2_score(y_train, y_train_pred)
print(r2_train)

# looking at the unique values of the ethnicity
print('In the train data, Ethnicity takes the three values as:',list(x_train['Ethnicity'].unique()))

print('In the train data, Ethnicity takes the three values as:',[(x_train.Ethnicity.unique())])  # doing this it gives array 

# separated trained and tested categorial values
nf=x_train.select_dtypes(include=['number']).columns.tolist()
cf=x_train.select_dtypes(exclude=['number']).columns.tolist()

# creating dummy variables for categorial 
x_train_design=pd.get_dummies(x_train[cf],drop_first=True)
x_test_design=pd.get_dummies(x_test[cf],drop_first=True)

'''
 drop_first le chai categorial ma 3 4 ota factors haru xa vane suruko factor lai chai drop gardinxa ra tesko predictions
aru 2 ota ko hot encoded 0 or 1 bata hunxa
'''
# combining both features now

x_train_design=pd.concat([x_train[nf],x_train_design],axis=1)
x_test_design= pd.concat([x_test[nf],x_test_design],axis=1)

x_train_design.head()

print(x_train_design.dtypes)
# esma bool auxa dtypes kinaki esma chai hamro columns haru lai true false value assign gareko hunxa pd.get_dummies le harek category ko factor lai ligera

# fiting model i.e. full model on design matrix including all numeric and categorial
model2=LinearRegression().fit(x_train_design,y_train)

# r2 score using .score()
train_score=model2.score(x_train_design,y_train)
test_score=model2.score(x_test_design,y_test)
print('Train R2: ',train_score)
print('Test R2: ', test_score)

#  r2 score calculate garda (y_pred,y_true) garne y value nai tara score use garda chai(x,y) hunxa

coef=pd.DataFrame(model2.coef_, index=x_train_design.columns, columns=['beta_value'])

# visualize crude measure of feature importance
sns.barplot(data=coef.T, orient='h').set(title="Model coefficients")

'''
here the index is used to assign the label to the rows in the dataframes which is the feature name of the predictor value
.T le chai coef matrix jun hamro normal dataframe huxna teslai transpose gardinxa i.e. hamro rows ma data variables hunxa normally jaslai chai
esle column ma change gardinxa which is the required one

'''
plt.show()

# fit a model to predict balance from income and student_yes

features=['Income','Student_Yes']
model3=LinearRegression().fit(x_train_design[features],y_train)

'''
here only feature cant be used because tyo vaneko list ho ani tesle euta matra predictor lai use garxa paxi value nikalda
so tyo full syntax le x_train_design bata 2 ota predictor ko respective data column in a 2D array format ma dinxa

'''
# collecting betas from model
B0=model3.intercept_
B1=model3.coef_[features.index('Income')]
B2=model3.coef_[features.index('Student_Yes')]

# displaying betas in dataframe
coefs=pd.DataFrame([B0,B1,B2],index=['Intercept']+features,columns=['Beta Values'])
print(coefs)

# visualize the model
sns.barplot(data=coefs.T, orient='v').set(title='Model Coefficients')
plt.show()

# explaining how the qualitative features affects the prediction line i.e. how the student and non student affects the income and balance
x_space=np.linspace(x['Income'].min(),x['Income'].max(),1000)  # creates 1000 number of points between maximum and min values of income and gives 1D array
y_yes=B0+B1*x_space+B2*1  # categorial features student is present

y_no=B0+B1*x_space+B2*0


# plotting data points
ax=sns.scatterplot(data=pd.concat([x_train_design, y_train],axis=1),x='Income',y='Balance',hue='Student_Yes',alpha=0.8)

# concat is done as plot garda y ra x axis ko same dataframe ma hunu parxa 

ax.plot(x_space, y_yes)
ax.plot(x_space, y_no)

plt.show()