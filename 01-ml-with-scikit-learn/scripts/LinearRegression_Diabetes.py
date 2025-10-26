# LinearRegression_Diabetes.py

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Explore the data
print("----- Head -----") #display from first to fifth lines in order to understand the data's shape
print(data.head())
print("\n----- Info -----") # display info about the columns, data, unmissed data
print(data.info())
print("\n----- Describe -----") # general statistics about the data like: mean, standard deviation, min, max, etc
print(data.describe())

# Split data into Features (X) and Target (y)
X = data.drop("Outcome", axis=1)  # All columns except 'target'
y = data["Outcome"]               # Target column

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n----- Model Performance -----")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
