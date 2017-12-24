import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data=pd.read_csv('challenge_dataset.txt',header=None)
data.columns=['a','b'] #somehow the pandas library does not allow the columns to be assigned at the time of reading



#Testing if data got imported correctly
'''print(data.head())
print(data['a'][:5])
print(data['b'][:5])

print(len(data['a']))
print(len(data['b']))'''

model=LinearRegression()

x=data['a']
y=data['b']

x,x_test,y,y_test=np.asarray(train_test_split(data['a'],data['b'], test_size=0.1))

x=x.values.reshape(-1,1) # Need to reshape the data for some reason. The model does not accept a list of values. It has to be a 2-D array
y=y.values.reshape(-1,1)

x_test=x_test.values.reshape(-1,1)
y_test=y_test.values.reshape(-1,1)

print(x)

model.fit(x,y)

print("Rsquared for the model:",model.score(x,y))

plt.scatter(x,y)
plt.plot(x,model.predict(x))
plt.show()

#Calculating the error of the new values

y_pred=model.predict(x_test)

print(len(y_pred),len(y_test))

print("mean squared error :",mean_squared_error(y_pred,y_test))



