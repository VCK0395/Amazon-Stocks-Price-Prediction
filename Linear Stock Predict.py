# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor






# Load Amazon's Stock Data
price = pd.read_csv('AMZN.csv')
last_price = price.tail(1)

# Drop the last Day
price = price.head(len(price)-1)


days = list()
adjusted_price = list()

# Picking the Data adn Adjusted Close Price
price_days = price.loc[:, "Date"]
price_adj = price.loc[:, "Adj Close"]
print(price_adj)

"""
# Append to a List
for D in price_days:
    days.append(D.split('-')[2])

for ADJ in price_adj:
    adjusted_price.append(float(ADJ))

# Prepare the Models
X = np.array(days).reshape(-1, 1)
y = adjusted_price


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SVR Model
SVR_model = SVR(kernel='rbf', C=1000, gamma=0.85)
SVR_model.fit(X_train, y_train)

# Linear Regression Model
Lin_model = LinearRegression()
Lin_model.fit(X_train, y_train)

# RandomForestRegressor Model
RFR_model = RandomForestRegressor()
RFR_model.fit(X_train, y_train)


# Test the Prediction 
last_day = [[31]]


print('LinearRegression Predicted Price =', Lin_model.predict(last_day))
print('SVR_rgf Predicted Price =', SVR_model.predict(last_day))
print('RFR Predicted Price =', RFR_model.predict(last_day))
print('Amazon Actual Price =', last_price["Adj Close"][19])
"""








