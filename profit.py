# Author: James Quirk
# Date: 02/20/2024
# Purpose: Test of basic machine learning model on superstore data

# Import libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Read as a csv using pandas library
data = pd.read_csv("superstore.csv")

# Declare y and X vector and matrices
cleanedData = data.dropna()
y = cleanedData ["Profit"]
X = cleanedData[["Postal Code", "Sales", "Quantity", "Discount"]]

# Separate training and test folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Do a linear regression with one command
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

# Use predict function to define the prediction
y_pred = linearModel.predict(X_test)

# Calculate MSE
print("MSE =", str(mean_squared_error(y_test,y_pred)))

# Ridge parameters
print("Learned Parameters:", linearModel.coef_)