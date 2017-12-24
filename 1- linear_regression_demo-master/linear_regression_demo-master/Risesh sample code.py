#Code to find relation between brainwts and bodywts. Creating and plotting the best fit line
#using linear regression, predicting the value of body wt based on brainwts

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#reading the fixed width formatted file

abc=pd.read_fwf('brain_body.txt')

print(abc.head())
# data reads correctly

brainwt=abc[['Brain']]
bodywt=abc[['Body']]

body_reg=linear_model.LinearRegression() #initializing the model 

body_reg.fit(brainwt,bodywt) #fitting the model using the x and y values

plt.scatter(brainwt,bodywt) #plottting the scatter plot with the x and y axis values
plt.plot(brainwt, body_reg.predict(brainwt)) #plotting a line chart with the predictions
plt.show()


# Here the p-value shows how statistically significant the results are. Need to know what is null hypothesis
# Null hypothesis is opposite of what you're trying to prove. So for example in this case, our null hypothesis will be that there is no relationship between the brain and body weights of animals. if p value is really low(less than 0.05), then it shows that the likelihood of the null hypothesis being true is very low. Thus the null hypothesis is disproved and the opposite is true. Meaning that there is a statistically significant relationship between the two. In this case, in y=mx+c, m has a non zero value.

##### There is no easy way to calculate p values for a linear regression model in python sklearn

#Another test would be the R squared, which is the goodness of fit. 
# Rsquared = 1 - (Squared Errors of Fitted line/Squared error of mean line).
# Here we want to see that the squared errors of the Fitted line should be much lower than the squared errors of the mean line through the data, which would show that our model is actually fitting the trend of the data.

print(body_reg.score(brainwt,bodywt))

#How good is a good Rsquare value. Depends from domain to domain. Essentiallly anything that is better than the mean, meaning R squared greater than 0.5 is somewhat better at modelling the data than the mean lines.


