
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# Create steps
steps = [
    ("imp_mean", SimpleImputer()),        # Step 1: Impute missing values
    ("scaler", StandardScaler()),         # Step 2: Scale features
    ("logreg", LogisticRegression())      # Step 3: Logistic Regression model
]

# Set up pipeline
pipeline = Pipeline(steps)

# Define parameters for GridSearchCV
params = {
    "logreg__solver": ["newton-cg", "saga", "lbfgs"],
    "logreg__C": np.linspace(0.001, 1.0, 10)
}

# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params, cv=5)
tuning.fit(X_train, y_train)

# Make predictions on test set
y_pred = tuning.predict(X_test)

# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {:.3f}".format(
    tuning.best_params_, accuracy_score(y_test, y_pred)
))
