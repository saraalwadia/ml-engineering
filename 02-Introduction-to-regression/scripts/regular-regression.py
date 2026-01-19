"""
Regularized Regression Types:

Ridge Regression (L2 Regularization) – يضيف مربع المعاملات إلى دالة التكلفة

Lasso Regression (L1 Regularization) – يضيف القيمة المطلقة للمعاملات → قد يضغط بعض المعاملات إلى صفر

Elastic Net – مزيج بين L1 و L2 → يقلل المعاملات ويختار الميزات المهمة
"""

# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score
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
# 2. Define K-Fold cross-validation
# ------------------------------
kf = KFold(n_splits=4, shuffle=True, random_state=42)

# ------------------------------
# 3. Define models with regularization
# ------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.5),
    "Elastic Net": ElasticNet(alpha=0.5, l1_ratio=0.5)
}

# RMSE scorer
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# ------------------------------
# 4. Evaluate models with cross-validation
# ------------------------------
for name, model in models.items():
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    rmse_scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer)
    
    print(f"--- {name} ---")
    print("R² scores per fold:", r2_scores)
    print("Average R²:", np.mean(r2_scores))
    print("RMSE per fold:", rmse_scores)
    print("Average RMSE:", np.mean(rmse_scores))
    print()
