import mlflow
from sklearn.linear_model import LogisticRegression

mlflow.set_experiment("Spam Classifier")

with mlflow.start_run():

    model = LogisticRegression(n_jobs=4)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", score)
    mlflow.log_param("n_jobs", 4)
    mlflow.log_artifact("train.py")