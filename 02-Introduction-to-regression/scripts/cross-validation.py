# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer

# ------------------------------
# 1. Prepare the data
# ------------------------------
# Features: [BMI, Age, Height]
X = np.array([
    [22, 25, 170],
    [24, 30, 165],
    [20, 22, 180],
    [28, 35, 160],
    [26, 28, 175],
    [23, 27, 168],
    [25, 32, 172],
    [21, 24, 178]
])

# Target variable: Weight
y = np.array([60, 65, 58, 70, 66, 62, 67, 59])

# ------------------------------
# 2. Define the model
# ------------------------------
model = LinearRegression()

# ------------------------------
# 3. Define K-Fold cross-validation
# ------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=42)  # 4 folds

# ------------------------------
# 4. Cross-validated R²
# ------------------------------
r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
print("R² scores for each fold:", r2_scores)
print("Average R²:", np.mean(r2_scores))

# ------------------------------
# 5. Cross-validated RMSE
# ------------------------------
# Define RMSE scorer
rmse_scorer = make_scorer(mean_squared_error, squared=False)  # squared=False -> RMSE

rmse_scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer)
print("RMSE for each fold:", rmse_scores)
print("Average RMSE:", np.mean(rmse_scores))
