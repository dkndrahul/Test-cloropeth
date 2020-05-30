import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

customers = pd.read_csv("Ecommerce_Customers")

# customers.head()
# customers.info()
# customers.describe()

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?
# sns.jointplot(x= customers['Time on Website'], y= customers['Yearly Amount Spent'], data=customers)

#** Do the same but with the Time on App column instead. **

# sns.jointplot(x= customers['Time on App'], y= customers['Yearly Amount Spent'], data=customers)

#** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# sns.jointplot(x=customers['Time on App'], y=customers['Length of Membership'], data=customers, kind='hex')

# Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.(Don't worry about the the colors)

# sns.pairplot(customers)

# *Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership. *

# sns.lmplot(y='Yearly Amount Spent', x='Length of Membership', data=customers)

#Training a Linear Regression Model
#Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to 
# train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only 
# has text info that the linear regression model can't use.

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
lm.fit(X_train, y_train)
print("Coefficients : \n",lm.coef_)

# Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# ** Use lm.predict() to predict off the X_test set of the data.**

predictions = lm.predict(X_test)
# y_predictions = lm.predict(y_test)
# print(predictions)
# plt.scatter(y_test, predictions)
# plt.xlabel("Y Test")
# plt.ylabel("Y Predictions")
# plt.show()

# Evaluating the Model
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).

# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. 
# Refer to the lecture or to Wikipedia for the formulas**

from sklearn import metrics
print("MAE: \n", metrics.mean_absolute_error(y_test, predictions))
print("MSE: \n", metrics.mean_squared_error(y_test, predictions))
print("RMSE: \n", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.distplot(y_test-predictions, bins=50)
# plt.show()

coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficients'])
print(coeff_df)