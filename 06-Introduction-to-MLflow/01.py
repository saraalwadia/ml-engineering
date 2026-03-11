# Import MLflow
import mlflow

# Create new experiment
mlflow.create_experiment("Unicorn Model")

# Tag new experiment
mlflow.set_experiment_tag("version", "1.0")

# Set the experiment
mlflow.set_experiment("Unicorn Model")
mlflow.set_experiment("My Experiment")
with mlflow.start_run():
    # model training here

# or without "with" directly use
mlflow.start_run()

# What we can register as logging
## only one metric
mlflow.log_metric("accuracy", 0.91)

## multiple metrics
mlflow.log_metrics({
    "accuracy": 0.91,
    "f1_score": 0.89
})

## log parameters
mlflow.log_param("learning_rate", 0.001) # log a single parameter

## log multiple parameters
mlflow.log_params({
    "learning_rate": 0.001,
    "n_estimators": 100
})
## log artifacts
mlflow.log_artifact("train.py") # log a single artifact
## log whole file artifacts
mlflow.log_artifacts("outputs/")