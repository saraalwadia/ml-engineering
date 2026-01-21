# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------
# 1. Prepare the data
# ------------------------------
# Columns: [BMI, Age, Height]
X = np.array([
    [22, 25, 170],
    [24, 30, 165],
    [20, 22, 180],
    [28, 35, 160],
    [26, 28, 175]
])

# Target variable: Weight
y = np.array([60, 65, 58, 70, 66])

# ------------------------------
# 2. Create and train the model
# ------------------------------
model = LinearRegression()
model.fit(X, y)

# ------------------------------
# 3. Print coefficients and intercept
# ------------------------------
print("Coefficients (a1, a2, a3):", model.coef_)
print("Intercept (b):", model.intercept_)

# ------------------------------
# 4. Make predictions for new data
# ------------------------------
# New example: BMI=23, Age=28, Height=168
X_new = np.array([[23, 28, 168]])
y_pred = model.predict(X_new)
print("Prediction for new data:", y_pred)
