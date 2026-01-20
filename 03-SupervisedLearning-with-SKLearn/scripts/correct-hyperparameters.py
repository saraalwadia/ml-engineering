# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------
# 1️⃣ Create a synthetic dataset for classification
# ---------------------------
# 0 = Legitimate, 1 = Fraudulent
X, y = make_classification(
    n_samples=500,        # number of samples
    n_features=5,         # number of features
    n_informative=3,      # number of informative features
    n_redundant=0,        # number of redundant features
    n_classes=2,           # binary classification
    weights=[0.7, 0.3],    # imbalanced dataset
    random_state=42
)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 2️⃣ Initialize the Random Forest model
# ---------------------------
rf = RandomForestClassifier(random_state=42)

# ---------------------------
# 3️⃣ Define a grid of hyperparameters to search
# ---------------------------
param_grid = {
    'n_estimators': [50, 100, 200],      # number of trees in the forest
    'max_depth': [5, 10, 15, None],      # maximum depth of each tree
    'min_samples_split': [2, 5, 10]      # minimum samples required to split a node
}

# ---------------------------
# 4️⃣ Set up Grid Search with 5-Fold Cross-Validation
# ---------------------------
# This step systematically tries all combinations of hyperparameters
# and uses cross-validation to evaluate their performance
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                # 5-Fold Cross-Validation
    scoring='f1',        # use F1-score because the dataset is imbalanced
    n_jobs=-1            # use all available CPU cores to speed up
)

# Fit the Grid Search to the training data
grid_search.fit(X_train, y_train)

# ---------------------------
# 5️⃣ Examine the best hyperparameters
# ---------------------------
# These are the hyperparameters that give the best cross-validated F1-score
print("Best Hyperparameters:", grid_search.best_params_)

# ---------------------------
# 6️⃣ Evaluate the model on the test set
# ---------------------------
best_rf = grid_search.best_estimator_    # retrieve the best model
y_pred = best_rf.predict(X_test)

# Print a detailed classification report
print(classification_report(y_test, y_pred))
