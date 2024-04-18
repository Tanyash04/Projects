# Baseball Performance Prediction

# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor

# 1. Load 'Train_C.csv'
train_data = pd.read_csv('Train_C.csv')

# Preprocessing
# Drop rows with NaN values
train_data = train_data.dropna()

# Feature Engineering
# Calculate hitting statistics
train_data['BA'] = train_data['H'] / train_data['AB']
train_data['OBP'] = (train_data['H'] + train_data['BB']) / (train_data['AB'] + train_data['BB'])
train_data['SLG'] = (train_data['H'] + train_data['2B'] + 2*train_data['3B'] + 3*train_data['HR']) / train_data['AB']

# Define features (X) and target variable (y)
X = train_data[['AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'BA', 'OBP', 'SLG']]
y = train_data['R']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=300, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Neural Network (Multi-layer Perceptron)
model_mlp = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_mlp.compile(optimizer='adam', loss='mean_squared_error')
model_mlp.fit(X_train_scaled, y_train, epochs=150, batch_size=32, validation_split=0.2)
mse_mlp = model_mlp.evaluate(X_test_scaled, y_test)

# Support Vector Regression (SVR)
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
mse_svr = mean_squared_error(y_test, y_pred_svr)

# K-Nearest Neighbors (KNN)
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
mse_knn = mean_squared_error(y_test, y_pred_knn)

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)

# Gaussian Process Regression
gpr = GaussianProcessRegressor()
gpr.fit(X_train_scaled, y_train)
y_pred_gpr = gpr.predict(X_test_scaled)
mse_gpr = mean_squared_error(y_test, y_pred_gpr)

# Decision Tree Regressor
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)

# Non-linear Regression
def feature_engineering(X):
    X_transformed = np.hstack((X, X**2))
    return X_transformed

def non_linear_regression(X, y):
    X_transformed = feature_engineering(X)
    theta = np.linalg.inv(X_transformed.T @ X_transformed) @ X_transformed.T @ y
    return theta

theta = non_linear_regression(X_train, y_train)
X_test_transformed = feature_engineering(X_test)
y_pred_non_linear = X_test_transformed @ theta
mse_non_linear = mean_squared_error(y_test, y_pred_non_linear)

# Display mean squared errors for each model
print("Mean Squared Errors:")
print("Random Forest:", mse_rf)
print("Neural Network (MLP):", mse_mlp)
print("SVR:", mse_svr)
print("KNN:", mse_knn)
print("Gradient Boosting:", mse_gbr)
print("Gaussian Process Regression:", mse_gpr)
print("Decision Trees:", mse_dt)
print("Non-linear Regression:", mse_non_linear)

# Plot the learning curve for Random Forest
estimator_range = range(1, 300)
train_errors_rf = []
test_errors_rf = []

for n_estimators in estimator_range:
    model_rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model_rf.fit(X_train, y_train)
    y_train_pred_rf = model_rf.predict(X_train)
    y_test_pred_rf = model_rf.predict(X_test)
    train_error_rf = mean_squared_error(y_train, y_train_pred_rf)
    test_error_rf = mean_squared_error(y_test, y_test_pred_rf)
    train_errors_rf.append(train_error_rf)
    test_errors_rf.append(test_error_rf)

plt.plot(estimator_range, train_errors_rf, label='Training Error (RF)')
plt.plot(estimator_range, test_errors_rf, label='Testing Error (RF)')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Squared Error')
plt.title('Random Forest Learning Curve')
plt.legend()
plt.show()

# Print the number of estimators with the lowest testing error for Random Forest
min_test_error_index_rf = np.argmin(test_errors_rf)
print("Number of estimators with lowest testing error (RF):", estimator_range[min_test_error_index_rf])

# Import necessary libraries for further analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Define non-linear feature engineering functions
def feature_engineering(X):
    X_transformed = np.hstack((X, X**2))
    return X_transformed

# Define non-linear regression model
def non_linear_regression(X, y):
    X_transformed = feature_engineering(X)
    theta = np.linalg.inv(X_transformed.T @ X_transformed) @ X_transformed.T @ y
    return theta

# Define a function to calculate mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Load the data
train_data = pd.read_csv('Train_C.csv')

# Preprocessing
train_data = train_data.dropna()

# Feature Engineering
train_data['BA'] = train_data['H'] / train_data['AB']
train_data['OBP'] = (train_data['H'] + train_data['BB']) / (train_data['AB'] + train_data['BB'])
train_data['SLG'] = (train_data['H'] + train_data['2B'] + 2*train_data['3B'] + 3*train_data['HR']) / train_data['AB']

# Define features and target variable
X = train_data[['AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'CS', 'BA', 'OBP', 'SLG']]
y = train_data['R']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the non-linear regression model
theta = non_linear_regression(X_train, y_train)

# Make predictions on the test set
X_test_transformed = feature_engineering(X_test)
y_pred_non_linear = X_test_transformed @ theta

# Calculate mean squared error
mse_non_linear = mean_squared_error(y_test, y_pred_non_linear)
print("Mean Squared Error (Non-linear Regression):", mse_non_linear)
