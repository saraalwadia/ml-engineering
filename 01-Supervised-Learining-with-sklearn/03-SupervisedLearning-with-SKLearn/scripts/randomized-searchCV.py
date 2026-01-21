# ---------------------------
# 1️⃣ Import libraries
# ---------------------------
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------
# 2️⃣ Create a synthetic dataset
# ---------------------------
# Generate a binary classification dataset
# 0 = Legitimate, 1 = Fraudulent
X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    weights=[0.7, 0.3],  # Imbalanced dataset
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 3️⃣ Initialize the Random Forest model
# ---------------------------
rf = RandomForestClassifier(random_state=42)

# ---------------------------
# 4️⃣ Define hyperparameter distributions
# ---------------------------
# These are the ranges from which RandomizedSearchCV will randomly sample
param_distributions = {
    'n_estimators': [50, 100, 200, 300],  # Number of trees
    'max_depth': [5, 10, 15, None],       # Maximum depth of each tree
    'min_samples_split': [2, 5, 10]       # Minimum samples required to split a node
}

# ---------------------------
# 5️⃣ Set up RandomizedSearchCV
# ---------------------------
# RandomizedSearchCV will randomly sample n_iter combinations from param_distributions
# It uses cross-validation (cv=5) to evaluate the performance of each combination
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=50,       # Number of random combinations to try
    cv=5,            # 5-Fold Cross-Validation
    scoring='f1',    # Use F1-score since the dataset is imbalanced
    n_jobs=-1,       # Use all CPU cores to speed up
    random_state=42  # Ensure reproducibility
)

# ---------------------------
# 6️⃣ Fit RandomizedSearchCV to the training data
# ---------------------------
random_search.fit(X_train, y_train)

# ---------------------------
# 7️⃣ Retrieve the best hyperparameters
# ---------------------------
print("Best Hyperparameters:", random_search.best_params_)

# ---------------------------
# 8️⃣ Evaluate the best model on the test set
# ---------------------------
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
