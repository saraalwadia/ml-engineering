# Load model from local filesystem
model = mlflow.sklearn.load_model("lr_local_v1")

# Training Data
X = df[["R&D Spend", "Administration", "Marketing Spend", "State"]]
y = df[["Profit"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

# Train Model
model.fit(X_train, y_train)

# Save model to local filesystem
mlflow.sklearn.save_model(model, "lr_local_v2")

# Log model to MLflow Tracking
mlflow.sklearn.log_model(lr_model, "lr_tracking")

# Get the last run
run = mlflow.last_active_run()

# Get the run_id of the above run
run_id = run.info.run_id

# Load model from MLflow Tracking
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lr_tracking")