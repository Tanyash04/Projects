# -*- coding: utf-8 -*-

# # 1. Load 'Train_C.csv'
import pandas as pd
train_data = pd.read_csv('add file path')

# Print the first five rows of the DataFrame
print(train_data.head())

# Drop rows with NaN values
train_data = train_data.dropna()

# Print the first five rows of the DataFrame
print(train_data.head())

"""Taking hitting statistics as input and calculating three additional statistics: Batting Average (BA), On-Base Percentage (OBP), and Slugging Percentage (SLG). These are common metrics used in baseball to evaluate the performance of hitters.

*   Calculating Batting Average (BA): Batting Average is calculated by dividing the total number of hits (H) by the total number of at-bats (AB) for each player.


*   Calculating On-Base Percentage (OBP): On-Base Percentage is calculated by dividing the sum of hits (H) and walks (BB) by the sum of at-bats (AB) and walks (BB) for each player.



*   Calculating Slugging Percentage (SLG): Slugging Percentage is calculated by dividing the total number of bases (H + 2B + 2*3B + 3*HR) by the total number of at-bats (AB) for each player.
"""

import pandas as pd

def calculate_hitting_statistics(train_data):
    train_data = train_data.copy()  # Creating a copy to avoid SettingWithCopyWarning

    # Batting Average (BA)
    train_data.loc[:, 'BA'] = train_data['H'] / train_data['AB']

    # On-Base Percentage (OBP)
    train_data.loc[:, 'OBP'] = (train_data['H'] + train_data['BB']) / (train_data['AB'] + train_data['BB'])

    # Slugging Percentage (SLG)
    train_data.loc[:, 'SLG'] = (train_data['H'] + train_data['2B'] + 2*train_data['3B'] + 3*train_data['HR']) / train_data['AB']

    return train_data


# Calculating hitting statistics
train_data = calculate_hitting_statistics(train_data)

# Print the DataFrame with hitting statistics
print(train_data)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features (X) and target variable (y)
X = train_data[['AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'BA', 'OBP', 'SLG']]
y = train_data['R']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np

#  non-linear feature engineering functions
def feature_engineering(X):
    # adding polynomial features (up to degree 2)
    X_transformed = np.hstack((X, X**2))
    return X_transformed

#adding polynomial features, such as squares of existing features, is to capture non-linear relationships between the input features and the target variable

#  non-linear regression model
def non_linear_regression(X, y):
    # Perform feature engineering
    X_transformed = feature_engineering(X)

    # Solve for theta using closed-form solution (normal equation)
    theta = np.linalg.inv(X_transformed.T @ X_transformed) @ X_transformed.T @ y

    return theta

#  function to calculate mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Training the non-linear regression model
theta = non_linear_regression(X_train, y_train)

# Making predictions on the test set
X_test_transformed = feature_engineering(X_test)
y_pred = X_test_transformed @ theta

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

print (theta)

# Load the test data from the CSV file
test_data = pd.read_csv('/content/test_set_labeled.csv')

# Preprocess the test data (calculate hitting statistics)
test_data = calculate_hitting_statistics(test_data)

# Extract features (X_test) and target variable (y_test) from the test data
X_test = test_data[['AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'BA', 'OBP', 'SLG']]
y_test = test_data['R']

# Making predictions on the test set
X_test_transformed = feature_engineering(X_test)
y_pred = X_test_transformed @ theta

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)
