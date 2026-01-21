# ============================================
# Example: Data Scaling in Scikit-Learn
# Using StandardScaler with Logistic Regression
# ============================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------
# 1. Create a simple dataset
# Features:
#   - Age (small scale)
#   - Salary (large scale)
# Target:
#   - Binary class (0 or 1)
# --------------------------------------------

X = np.array([
    [20, 2000],
    [25, 2500],
    [30, 3000],
    [35, 4000],
    [40, 5000]
])

y = np.array([0, 0, 1, 1, 1])

# --------------------------------------------
# 2. Split the dataset into training and test sets
# IMPORTANT:
#   - Always split BEFORE scaling
#   - This avoids data leakage
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# 3. Initialize the scaler
# StandardScaler:
#   - Centers data (mean = 0)
#   - Scales data (std = 1)
# --------------------------------------------

scaler = StandardScaler()

# --------------------------------------------
# 4. Fit the scaler on TRAINING data only
# Then transform both training and test data
# --------------------------------------------

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# 5. Train a supervised learning model
# Logistic Regression is sensitive to feature scale
# --------------------------------------------

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --------------------------------------------
# 6. Make predictions on the test set
# --------------------------------------------

y_pred = model.predict(X_test_scaled)

# --------------------------------------------
# 7. Evaluate the model
# --------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# --------------------------------------------
# Notes:
# - Scaling improves model performance and stability
# - Never fit the scaler on test data
# - Tree-based models do NOT need scaling
# - Linear models, SVM, KNN DO need scaling
# --------------------------------------------
