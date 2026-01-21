import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=0.1),
    "Lasso": Lasso(alpha=0.1)
}

results = []

# Loop through the models
for model in models.values():  # iterate over the model objects
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
    
    # Append the results
    results.append(cv_scores)

# Create a box plot of the results
plt.boxplot(results, labels=models.keys())  # use models.keys() directly
plt.ylabel("R^2 scores")
plt.title("Cross-Validation Results for Regression Models")
plt.show()
