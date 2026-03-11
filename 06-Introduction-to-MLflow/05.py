import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

mlflow.sklearn.save_model(model, "local_model")

mlflow.sklearn.log_model(model, "tracking_model")

run = mlflow.last_active_run()
run_id = run.info.run_id

loaded_model = mlflow.sklearn.load_model(
    f"runs:/{run_id}/tracking_model"
)