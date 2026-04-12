# Set the MLflow Projects run method
model_evaluation = mlflow.projects.run(
    uri="./",
    # Set the entry point to model_evaluation
    entry_point="model_evaluation",
    # Set the parameter run_id to the run_id output of previous step
    parameters={
        "run_id": model_engineering_run_id,
    },
    env_manager="local"
)

print(model_evaluation.get_status())