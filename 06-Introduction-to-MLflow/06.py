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