# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create pipeline steps
steps = [("scaler", StandardScaler()),
         ("lasso", Lasso(alpha=0.5))]

# Instantiate the pipeline
pipeline =  Pipeline(steps)
pipeline.fit(X_train, y_train)

# Calculate and print R-squared
print(pipeline.score(X_test, y_test))

# Build the steps
steps = [
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression())
]

pipeline = Pipeline(steps)

# Create the parameter space
parameters = {
    "logreg__C": np.linspace(0.001, 1.0, 20)
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21
)

# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training data
cv.fit(X_train, y_train)

print(cv.best_score_, "\n", cv.best_params_)
