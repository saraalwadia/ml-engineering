# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ------------------------------
# 1. Prepare the data
# ------------------------------
# BMI values (feature) - reshape to 2D array for scikit-learn
X_bmi = np.array([18.5, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]).reshape(-1, 1)

# Weight values (target)
y_weight = np.array([55, 58, 60, 64, 68, 72, 75])

# ------------------------------
# 2. Create and train the model
# ------------------------------
model = LinearRegression()  # Initialize the Linear Regression model
model.fit(X_bmi, y_weight)  # Train the model on the data

# ------------------------------
# 3. Print the slope and intercept
# ------------------------------
print("Slope (b1):", model.coef_[0])       # How much weight changes per unit BMI
print("Intercept (b0):", model.intercept_) # Starting value of weight when BMI=0

# ------------------------------
# 4. Make predictions for new BMI values
# ------------------------------
X_new = np.array([21, 25, 29]).reshape(-1, 1)  # New BMI values
y_pred = model.predict(X_new)                   # Predicted weights
print("Predictions for new BMI values:", y_pred)

# ------------------------------
# 5. Visualize the data and regression line
# ------------------------------
plt.scatter(X_bmi, y_weight, color='blue', label='Data points')       # Actual data
plt.plot(X_bmi, model.predict(X_bmi), color='red', label='Regression line')  # Regression line

plt.xlabel('BMI')     # x-axis label
plt.ylabel('Weight')  # y-axis label
plt.title('Linear Regression: Weight vs BMI')  # plot title
plt.legend()
plt.show()
