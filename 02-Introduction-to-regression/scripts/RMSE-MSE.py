from sklearn.metrics import mean_squared_error
import numpy as np

# True values
y_true = np.array([60, 65, 58, 70])

# Predicted values
y_pred = np.array([61, 64, 59, 68])

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)
print("MSE:", mse)

# Calculate RMSE
rmse = np.sqrt(mse)
print("RMSE:", rmse)
