from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 1 Create a sample dataset for Blood Glucose Levels
# Features: Age, BMI, Exercise_hours_per_week
# Target: Blood_Glucose_Level (mg/dL)
data = {
    'Age': [25, 30, 45, 50, 35, 40, 60, 55],
    'BMI': [22, 28, 26, 30, 24, 27, 31, 29],
    'Exercise_hours': [5, 2, 1, 0, 4, 2, 0, 1],
    'Blood_Glucose_Level': [90, 110, 150, 160, 100, 130, 180, 170]
}

df = pd.DataFrame(data)

# 2 Split data into features (X) and target (y)
X = df[['Age', 'BMI', 'Exercise_hours']]
y = df['Blood_Glucose_Level']

# 3 Split data into Training and Test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4 Standardize features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5 Create KNN Regressor model with k=3
knn_reg = KNeighborsRegressor(n_neighbors=3)

# 6 Train the model
knn_reg.fit(X_train, y_train)

# 7 Predict on test data
y_pred = knn_reg.predict(X_test)

# 8 Evaluate the model
mse = mean_squared_error(y_test, y_pred)   # Mean Squared Error
r2 = r2_score(y_test, y_pred)             # R² Score (coefficient of determination)

print("Predicted Blood Glucose Levels:", y_pred)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
